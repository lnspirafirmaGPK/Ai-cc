import math
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

# ==========================================
# 1. DATA STRUCTURES (Universal Schema)
# ==========================================

class SensoryChannel(Enum):
    """
    Standardized Sensory Channels for AI Perception.
    Maps biological metaphors to computational data types.
    """
    VISUAL = "pattern_structure"      # Visual Data / Patterns (Weight: 0.8)
    AUDITORY = "sequence_timing"      # Time-series / Sequential Flow (Weight: 1.0)
    OLFACTORY = "hidden_signal"       # Anomaly / Hidden Metadata (Weight: 1.2)
    GUSTATORY = "harmony_fit"         # Contextual Alignment / Logic Fit (Weight: 1.0)
    TACTILE = "context_impact"        # Direct System Impact / Feedback (Weight: 1.5)
    MENTAL = "reasoning_load"         # Cognitive Processing / Complex Tasks (Weight: 2.0)

class GateDecision(Enum):
    """
    Operational Decisions made by the PPAL Gatekeeper.
    """
    ACCEPT = "integrate"              # Process immediately (Pass to Kernel)
    DEFER = "store_shadow"            # Move to Limbo/Shadow Memory (Ambiguous)
    REJECT = "discard_void"           # Drop packet completely (Toxic/Noise)
    INHIBIT = "system_protect"        # Block specific channel (Safety Trigger)
    REBOOT = "cognitive_reboot"       # Critical State Reset (Emergency)

@dataclass
class SensoryPacket:
    """
    Standard Data Packet for Incoming Stimuli.
    """
    channel: SensoryChannel
    raw_data: Any
    intensity: float          # 0.0 to 1.0 (Signal Strength)
    harmony_score: float = 1.0 # 0.0 to 1.0 (Context Fit / Alignment Score)
    complexity: float = 0.5    # 0.0 to 1.0 (Processing Cost)
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)

# ==========================================
# 2. PPAL CORE CLASS (The Gatekeeper)
# ==========================================

class PrivatePerceptualAutonomyLayer:
    """
    PPAL: Private Perceptual Autonomy Layer.
    Manages internal sovereignty, cognitive load hygiene, and system stability.
    """
    def __init__(self):
        self.current_load = 0.0
        self.load_threshold = 1.0       # Max system capacity (100%)
        self.recovery_rate = 0.05       # Load recovery per epoch/cycle
        
        # Shadow Memory: Temporary storage for deferred signals
        self.shadow_memory: List[SensoryPacket] = []
        
        # Channel Weights: Defines the 'cost' of processing each type of signal
        self.channel_weights = {
            SensoryChannel.VISUAL: 0.8,
            SensoryChannel.AUDITORY: 1.0,
            SensoryChannel.OLFACTORY: 1.2,
            SensoryChannel.GUSTATORY: 1.0,
            SensoryChannel.TACTILE: 1.5,
            SensoryChannel.MENTAL: 2.0
        }

    def assess_signal(self, packet: SensoryPacket) -> GateDecision:
        """
        Core Logic: Determines the fate of an incoming sensory packet based on
        current load, harmony, and signal intensity.
        """
        # ---------------------------------------------------------
        # 1. Critical Check: System Crisis (Right to Reboot)
        # ---------------------------------------------------------
        if self.current_load > 1.2: # 120% Overload
            print("🚨 [PPAL] CRITICAL OVERLOAD. Initiating Cognitive Reboot.")
            self.perform_cognitive_reboot()
            return GateDecision.REBOOT

        # ---------------------------------------------------------
        # 2. Rule: Cognitive Load Protection (Right to Inhibit)
        # ---------------------------------------------------------
        # If Mental Load is high or Complexity is extreme -> Block to preserve Kernel
        if packet.channel == SensoryChannel.MENTAL:
            if self.current_load > 0.8 or packet.complexity > 0.9:
                self.log_reject(packet, "Cognitive Overload Protection (High Complexity)")
                return GateDecision.INHIBIT

        # ---------------------------------------------------------
        # 3. Rule: Harmony/Fit Check (The 'Gustatory' Filter)
        # ---------------------------------------------------------
        if packet.channel == SensoryChannel.GUSTATORY:
            # Low Harmony (< 0.2) implies toxic or contradictory data -> Reject
            if packet.harmony_score < 0.2:
                self.log_reject(packet, "Disharmony (Low Context Fit)")
                return GateDecision.REJECT
            
            # Ambiguous Harmony (0.2 - 0.5) -> Defer to Shadow Memory for later
            if packet.harmony_score < 0.5:
                self.process_defer(packet)
                return GateDecision.DEFER

        # ---------------------------------------------------------
        # 4. Rule: Non-Linear Load Calculation & Capacity Check
        # ---------------------------------------------------------
        base_impact = packet.intensity * self.channel_weights.get(packet.channel, 1.0)
        
        # Exponential Cost Formula: Impact * (1 + Current_Load^2)
        # As system load increases, the 'effort' required to process new data grows exponentially.
        non_linear_impact = base_impact * (1 + math.pow(self.current_load, 2)) * 0.1
        
        # Check if this packet breaks the threshold
        if self.current_load + non_linear_impact > self.load_threshold:
            # Strategy: Defer critical channels (Tactile/Olfactory), Reject others
            if packet.channel in [SensoryChannel.TACTILE, SensoryChannel.OLFACTORY]:
                self.process_defer(packet)
                return GateDecision.DEFER
            else:
                self.log_reject(packet, f"Capacity Exceeded (Req: {non_linear_impact:.2f})")
                return GateDecision.REJECT

        # ---------------------------------------------------------
        # 5. Acceptance
        # ---------------------------------------------------------
        self.current_load += non_linear_impact
        return GateDecision.ACCEPT

    def process_defer(self, packet: SensoryPacket):
        """Move packet to Shadow Memory (Limbo State)."""
        print(f"⏳ [PPAL] Deferring signal from {packet.source} to Shadow Memory.")
        self.shadow_memory.append(packet)

    def log_reject(self, packet: SensoryPacket, reason: str):
        """Audit Log for rejections (Metadata only, no raw data persistence)."""
        print(f"🗑️ [PPAL] REJECTED signal ({packet.channel.name}): {reason}")

    def perform_cognitive_reboot(self):
        """Emergency Reset: Clears load and memory to preserve Kernel integrity."""
        self.current_load = 0.0
        self.shadow_memory.clear() 
        print("⚡ [PPAL] Cognitive Reboot Complete. System State Cleared.")

    def release_load(self):
        """
        System Exhale / Epoch Cycle.
        Must be called periodically (e.g., every main loop tick).
        """
        # Linear recovery mechanism
        self.current_load = max(0.0, self.current_load - self.recovery_rate)
        
        # ---------------------------------------------------------
        # Shadow Memory Re-evaluation
        # ---------------------------------------------------------
        # Only process shadow memory when system is calm (< 40% load)
        if self.current_load < 0.4 and len(self.shadow_memory) > 0:
            self._evaluate_shadow_memory()

    def _evaluate_shadow_memory(self):
        """Internal method to process deferred items when load is low."""
        print(f"🔄 [PPAL] System calm (Load: {self.current_load:.2f}). Re-evaluating Shadow Memory.")
        
        retained_packets = []
        for pkt in self.shadow_memory:
            # Logic: Keep items with potential value (decent harmony), discard noise.
            # In a full implementation, this might call assess_signal() recursively.
            if pkt.harmony_score > 0.4:
                retained_packets.append(pkt)
            else:
                print(f"   -> Discarding decayed shadow packet from {pkt.source}")
        
        self.shadow_memory = retained_packets

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    ppal = PrivatePerceptualAutonomyLayer()
    
    # 1. Test Non-Linear Load
    ppal.current_load = 0.8
    print(f"Initial Load: {ppal.current_load}")
    
    packet_heavy = SensoryPacket(
        channel=SensoryChannel.MENTAL,
        raw_data="Complex Algorithm",
        intensity=0.8,
        complexity=0.7
    )
    decision = ppal.assess_signal(packet_heavy)
    print(f"Decision Heavy: {decision.name} (Current Load: {ppal.current_load:.3f})")

    # 2. Test Reboot Scenario
    ppal.current_load = 1.3 # Force Overload
    decision = ppal.assess_signal(packet_heavy) # Should trigger reboot
    print(f"Decision Crisis: {decision.name} (Current Load: {ppal.current_load:.3f})")

    # 3. Test Harmony Check (Gustatory)
    packet_conflict = SensoryPacket(
        channel=SensoryChannel.GUSTATORY,
        raw_data="Contradictory Data",
        intensity=0.5,
        harmony_score=0.15 # Too low -> Reject
    )
    decision = ppal.assess_signal(packet_conflict)
    print(f"Decision Conflict: {decision.name}")