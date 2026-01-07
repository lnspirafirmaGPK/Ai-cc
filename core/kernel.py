# lnspirafirmagpk/ai-cc/Ai-cc-main/core/kernel.py
from core.wisdom.vault import WisdomVault

class Kernel:
    def __init__(self):
        self.vault = WisdomVault()

    def process_input(self, user_input):
        # ตรรกะการใช้เหตุผล (Reasoning) ของคุณเอง
        context = self.vault.retrieve(user_input)
        return context
