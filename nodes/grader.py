from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from nodes.state import GraphState
from utils.graph_tracer import graph_tracer

def initialize_graders():
    """Initialize grader chains for document relevance, hallucination checking, and answer quality."""
    llm = AzureChatOpenAI(deployment_name="gpt-4-2")
    
    # Retrieval grader
    retrieval_grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question. 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            give the answer in single word 'yes' or 'no'"""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = retrieval_grader_prompt | llm
    
    # Question rewriter
    question_rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", """You a question re-writer that converts an input question to a better version that is optimized
             for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
             return question only. documents stored are in markdown format so form query that is more semantically similar in the markdown format."""),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question such that it can be used for sematic document retrival."),
    ])
    question_rewriter = question_rewriter_prompt | llm | StrOutputParser()
    
    # Hallucination grader
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
             give the answer in single word 'yes' or 'no'"""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | llm
    
    # Answer grader
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an answer addresses / resolves a question 
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
             give the answer in single word 'yes' or 'no'"""),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_grader = answer_prompt | llm
    
    return {
        "retrieval_grader": retrieval_grader,
        "question_rewriter": question_rewriter,
        "hallucination_grader": hallucination_grader,
        "answer_grader": answer_grader
    }

# Global graders
graders = None

def get_graders():
    """Get or initialize graders."""
    global graders
    if graders is None:
        graders = initialize_graders()
    return graders

def grade_documents(state: GraphState) -> GraphState:
    """Determines whether the retrieved documents are relevant to the question."""

    graph_tracer.add_trace("grade_documents", state)
    
    question = state["question"]
    documents = state["documents"]
    
    # Get graders
    graders = get_graders()
    retrieval_grader = graders["retrieval_grader"]

    # Scoring each doc
    filtered_docs = []
    for i, d in enumerate(documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.content
        if grade == "yes":
            filtered_docs.append(d)
    
    # Creating updated state        
    updated_state = {"documents": filtered_docs, "question": question}
    
    # Adding trace with filtering results
    graph_tracer.add_trace("grade_documents", updated_state, 
                          decision=f"Filtered {len(documents)} docs to {len(filtered_docs)} relevant docs")
    
    return updated_state

def transform_query(state: GraphState) -> GraphState:
    """Transform the query to produce a better question."""

    graph_tracer.add_trace("transform_query", state)
    
    question = state["question"]
    documents = state["documents"]
    
    # Getting graders
    graders = get_graders()
    question_rewriter = graders["question_rewriter"]

    # Rewriting question
    better_question = question_rewriter.invoke({"question": question})
    
    # Creating updated state
    updated_state = {"documents": documents, "question": better_question}
    
    # Adding trace with query transformation
    graph_tracer.add_trace("transform_query", updated_state, 
                          decision=f"Transformed query: '{question}' -> '{better_question}'")
    
    return updated_state

def decide_to_generate(state: GraphState) -> str:
    """Determines whether to generate an answer, or re-generate a question."""

    graph_tracer.add_trace("decide_to_generate", state)
    
    filtered_documents = state["documents"]
    decision = None

    if not filtered_documents:
        # All documents have been filtered out
        decision = "transform_query"
        graph_tracer.add_trace("decide_to_generate", state, decision="No relevant docs, transforming query")
    else:
        # We have relevant documents, so generating answer
        decision = "generate"
        graph_tracer.add_trace("decide_to_generate", state, 
                              decision=f"Found {len(filtered_documents)} relevant docs, generating answer")

    return decision

def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """Determines whether the generation is grounded in the document and answers question."""

    graph_tracer.add_trace("grade_generation", state)
    
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    # Getting graders
    graders = get_graders()
    hallucination_grader = graders["hallucination_grader"]
    answer_grader = graders["answer_grader"]

    # Checking hallucination
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.content
    decision = None

    if grade == "yes":
        # Checking question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.content
        if grade == "yes":
            decision = "useful"
            graph_tracer.add_trace("grade_generation", state, 
                                  decision="Generation is grounded and answers question")
        else:
            decision = "not useful"
            graph_tracer.add_trace("grade_generation", state, 
                                  decision="Generation is grounded but doesn't answer question")
    else:
        decision = "not supported"
        graph_tracer.add_trace("grade_generation", state, 
                              decision="Generation contains hallucinations")

    return decision 