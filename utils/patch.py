import os
import streamlit.watcher.local_sources_watcher

def apply_streamlit_patches():
    original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths

    def patched_get_module_paths(module):
        try:
            module_name = getattr(module, "__name__", "")
            if "torch._classes" in module_name or "torch.classes" in module_name:
                return []
            
            module_type = str(type(module))
            if "CythonDotParallel" in module_type:
                return []
        except Exception:
            return []
            
        return original_get_module_paths(module)

    streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

    os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false" 