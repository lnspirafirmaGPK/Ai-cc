# เพิ่มลงใน lnspirafirmagpk/ai-cc/Ai-cc-main/platform/api/server.py
from core.kernel import Kernel

class OrolarReasoningProvider(ReasoningProvider):
    """สมอง AI ที่คุณออกแบบเอง"""
    def __init__(self):
        self.kernel = Kernel()

    async def reason(self, prompt: str) -> str:
        # ให้ Kernel ประมวลผลแทนการส่งไป API ภายนอก
        answer = self.kernel.process_input(prompt)
        return answer

# เปลี่ยนการเรียกใช้ Provider
reasoning_provider = OrolarReasoningProvider()
