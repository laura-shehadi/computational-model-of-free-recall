import random
import matplotlib.pyplot as plt

from pyClarion import Agent, Input, Choice, Pool, Event, ChunkStore, NumDict, ks_crawl
from pyClarion.components.rules import RuleStore
from pyClarion.knowledge import (
    Root, ChunkFamily, RuleFamily, DataFamily,
    AtomFamily, BusFamily, Rule, Atoms, Atom, Buses, Bus
)
from pyClarion.knowledge.terms import this


class MainBuses(Buses):
    wm: Bus


class ModelLayout(BusFamily):
    main: MainBuses


class Phase(Atoms):
    study: Atom
    interference: Atom
    recall: Atom
    stop: Atom


class ModelData(DataFamily):
    phase: Phase


class ModelKeyspace[D: DataFamily](Root):
    b: ModelLayout
    c: ChunkFamily
    p: AtomFamily
    r: RuleFamily
    d: D

    def __init__(self, data_type: type[D]) -> None:
        super().__init__()
        self.d = self["d"] = data_type()


class SerialPositionAgent[D: DataFamily](Agent):
    ks: ModelKeyspace[D]

    def __init__(self, name: str, data_type: type[D], phase: str = "phase", f: float = 1):
        ks = ModelKeyspace(data_type)
        phase_sort = ks.d[phase]
        assert isinstance(phase_sort, Atoms)

        super().__init__(name, ks)
        self.ks = ks

        with self:
            self.lhs = ChunkStore(f"{name}.lhs", ks.c, (ks.b.main, ks.d))
            self.rhs = ChunkStore(f"{name}.rhs", ks.c, (ks.b.main, ks.d))
            self.ps = RuleStore(f"{name}.ps", ks.d, self.lhs.c, self.rhs.c)

            self.input = Input(f"{name}.input", (ks.b.main, ks.d))
            self.phase = Choice(
                f"{name}.phase",
                p=ks.p,
                s=ks.d,
                d=(ks.b.main.wm, phase_sort),
                sd=1e-3,
                f=f
            )
            self.pool_in = Pool(
                f"{name}.pool_in",
                p=ks.p,
                d=(ks.b.main, ks.d),
                agg=NumDict.sum
            )
            self.lhs_bu = self.lhs.bottom_up(f"{name}.lhs_bu")
            self.lhs_layer = self.ps.lhs_layer(f"{name}.lhs_layer")
            self.rule_selector = Choice(
                f"{name}.rule_selector",
                p=ks.p,
                s=ks.d,
                d=self.ps.r,
                sd=1e-3
            )
            self.rhs_layer = self.ps.rhs_layer(f"{name}.rhs_layer")
            self.rhs_td = self.rhs.top_down(f"{name}.rhs_td")

        self.phase = (
            (self.input, self.phase)
            >> self.pool_in
            >> self.lhs_bu
            >> self.lhs_layer
            >> self.rule_selector
            >> self.rhs_layer
            >> self.rhs_td
            >> self.phase
        )

    def resolve(self, event: Event) -> None:
        if event.source == self.input.send:
            self.system.schedule(self.pool_in.forward())
        if event.source == self.lhs_layer.forward:
            self.system.schedule(self.rule_selector.trigger())
        if event.source == self.rhs_td.forward:
            self.system.schedule(self.phase.trigger())


def make_phase_rules(ks: ModelKeyspace[ModelData]) -> list[Rule]:
    b, d = ks.b, ks.d
    return [
        "to_interference" ^
        + b.main.wm ** d.phase.study
        - b.main.wm ** this.rule
        >>
        + b.main.wm ** d.phase.interference,

        "to_recall" ^
        + b.main.wm ** d.phase.interference
        - b.main.wm ** this.rule
        >>
        + b.main.wm ** d.phase.recall,

        "to_stop" ^
        + b.main.wm ** d.phase.recall
        - b.main.wm ** this.rule
        >>
        + b.main.wm ** d.phase.stop
    ]


# -----------------------------
# Memory simulation model
# -----------------------------
class SerialPositionMemoryModel:
    def __init__(
        self,
        list_length=12,
        stm_capacity=7,
        init_stm=1.0,
        stm_decay=0.12,
        distract_decay=0.18,
        rehearsal_gain=0.05,
        ltm_gain=0.035,
        primacy_boost=0.04,
        recall_threshold=0.18,
        recall_noise=0.06,
        distract_steps=2,
    ):
        self.list_length = list_length
        self.stm_capacity = stm_capacity
        self.init_stm = init_stm
        self.stm_decay = stm_decay
        self.distract_decay = distract_decay
        self.rehearsal_gain = rehearsal_gain
        self.ltm_gain = ltm_gain
        self.primacy_boost = primacy_boost
        self.recall_threshold = recall_threshold
        self.recall_noise = recall_noise
        self.distract_steps = distract_steps

    def run_trial(self, verbose=False):
        items = list(range(self.list_length))

        stm = {i: 0.0 for i in items}
        ltm = {i: 0.0 for i in items}
        rehearsals = {i: 0 for i in items}
        recalled = {i: False for i in items}

        # this is the STM buffer: last few items presented
        stm_buffer = []

        # -----------------------------
        # STUDY PHASE
        # -----------------------------
        if verbose:
            print("\n--- STUDY PHASE ---")

        for pos, item in enumerate(items):
            # decay all current STM activations
            for j in items:
                if stm[j] > 0:
                    stm[j] = max(0.0, stm[j] - self.stm_decay)

            # add new item to STM buffer
            stm_buffer.append(item)
            if len(stm_buffer) > self.stm_capacity:
                dropped = stm_buffer.pop(0)
                stm[dropped] = 0.0

            # new item gets strong STM activation
            stm[item] = self.init_stm

            # rehearse all items still in STM buffer
            for j in stm_buffer:
                stm[j] += self.rehearsal_gain
                rehearsals[j] += 1
                ltm[j] += self.ltm_gain

                # small primacy advantage for earliest list positions
                if pos < self.stm_capacity:
                    serial_pos = j + 1
                    ltm[j] += self.primacy_boost * (1 / serial_pos)

            if verbose:
                print(f"Presented item {item+1}")
                print(" STM buffer:", [x + 1 for x in stm_buffer])
                print(" STM:", {k+1: round(v, 3) for k, v in stm.items() if v > 0})
                print(" LTM:", {k+1: round(v, 3) for k, v in ltm.items() if v > 0})

        # -----------------------------
        # INTERFERENCE PHASE
        # -----------------------------
        if verbose:
            print("\n--- INTERFERENCE PHASE ---")

        for step in range(self.distract_steps):
            for j in items:
                stm[j] = max(0.0, stm[j] - self.distract_decay)

            if verbose:
                print(f"Interference step {step+1}")
                print(" STM:", {k+1: round(v, 3) for k, v in stm.items() if v > 0})

        # -----------------------------
        # RECALL PHASE
        # -----------------------------
        if verbose:
            print("\n--- RECALL PHASE ---")

        recalled_order = []

        while True:
            candidates = []

            for j in items:
                if not recalled[j]:
                    total_activation = ltm[j] + stm[j] + random.gauss(0, self.recall_noise)
                    candidates.append((j, total_activation))

            if not candidates:
                break

            best_item, best_act = max(candidates, key=lambda x: x[1])

            if verbose:
                shown = {j+1: round(a, 3) for j, a in candidates}
                print(" Candidate activations:", shown)
                print(f" Best item = {best_item+1}, activation = {best_act:.3f}")

            # soft probabilistic recall instead of hard threshold
            recall_prob = 1 / (1 + pow(2.71828, -5 * (best_act - self.recall_threshold)))

            if random.random() > recall_prob:
                if verbose:
                    print(" Recall stopped probabilistically.")
                break

            recalled[best_item] = True
            recalled_order.append(best_item)

            stm[best_item] = 0.0
            ltm[best_item] *= 0.6

            if verbose:
                print(f" Recalled item {best_item+1}")

        return recalled_order, stm, ltm, rehearsals

    def run_experiment(self, n_trials=500):
        recall_counts = [0] * self.list_length

        for _ in range(n_trials):
            recalled_order, _, _, _ = self.run_trial(verbose=False)
            recalled_set = set(recalled_order)

            for pos in range(self.list_length):
                if pos in recalled_set:
                    recall_counts[pos] += 1

        return [count / n_trials for count in recall_counts]
    

# -----------------------------
# Create and run pyClarion controller
# -----------------------------
agent = SerialPositionAgent("serial_position_agent", ModelData)

encoding_event = agent.ps.encode(*make_phase_rules(agent.ks))
agent.system.schedule(encoding_event)
agent.system.run_all()

init = + agent.ks.b.main.wm ** agent.ks.d.phase.study
agent.system.schedule(agent.input.send(init))

print("=== pyClarion phase transitions ===")
while agent.system.queue:
    event = agent.system.advance()
    if event.source == agent.rule_selector.select:
        key = agent.rule_selector.poll()[~agent.ps.r]
        print(event.describe())
        print(ks_crawl(agent.ks, key))


# -----------------------------
# Create and run memory model
# -----------------------------
model = SerialPositionMemoryModel(
    list_length=12,
    stm_capacity=7,
    init_stm=0.75,
    stm_decay=0.12,
    distract_decay=0.15,
    rehearsal_gain=0.05,
    ltm_gain=0.035,
    primacy_boost=0.045,
    recall_threshold=0.08,
    recall_noise=0.06,
    distract_steps=3,
)

recall_probs = model.run_experiment(n_trials=10000)

print("Recall probability by serial position:")
for i, p in enumerate(recall_probs, start=1):
    print(f"Position {i}: {p:.3f}")

positions = list(range(1, len(recall_probs) + 1))
plt.figure(figsize=(8, 5))
plt.plot(positions, recall_probs, marker="o")
plt.xlabel("Serial Position")
plt.ylabel("Recall Probability")
plt.title("Simulated Serial Position Curve")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
