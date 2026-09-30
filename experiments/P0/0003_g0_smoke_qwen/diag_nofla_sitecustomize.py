# RUN-0003 diagnostic only (not a fix): make `import fla` fail in this process tree so transformers binds its torch GDN fallback.
import sys
sys.modules['fla'] = None
