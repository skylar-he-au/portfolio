"""Post-deploy browser patrol (SPEC: 权限靠代码锁死, 部署后巡检).

Telemetry off before browser-use is ever imported: the workflow sets these too, this is the
second time (browser-use reads ANONYMIZED_TELEMETRY when it starts; BH_TELEMETRY is its
browser-harness dependency's). The update check is CLI-only in 1.0.6; off anyway.
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["BH_TELEMETRY"] = "0"
os.environ["BROWSER_USE_NO_UPDATE_CHECK"] = "1"
