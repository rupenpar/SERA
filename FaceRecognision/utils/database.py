import importlib
from typing import Dict, Any, Optional


class CloudDatabase:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        members_table: str = "premium_members",
        logs_table: str = "entry_logs",
    ) -> None:
        self.members_table = members_table
        self.logs_table = logs_table
        self.enabled = (
            bool(supabase_url)
            and bool(supabase_key)
            and "YOUR-PROJECT" not in supabase_url
            and "YOUR_SUPABASE" not in supabase_key
        )
        self.client: Optional[Any] = None

        if self.enabled:
            try:
                supabase_module = importlib.import_module("supabase")
                self.client = supabase_module.create_client(supabase_url, supabase_key)
            except Exception:
                self.enabled = False

    def get_member_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.client or not name or name == "Unknown":
            return None

        try:
            response = (
                self.client.table(self.members_table)
                .select("member_id,name,premium_tier,membership_status")
                .ilike("name", name)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
            return None
        except Exception:
            return None

    def log_access_attempt(self, data: Dict[str, Any]) -> bool:
        if not self.enabled or not self.client:
            return False

        try:
            self.client.table(self.logs_table).insert(data).execute()
            return True
        except Exception:
            return False
