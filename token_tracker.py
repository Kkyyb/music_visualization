"""Token usage tracker."""
from __future__ import annotations

from config import PRICE_FLUX2_PRO, PRICE_SAM3D_EACH, PRICE_TABLE
from state import TokenUsage


class TokenTracker:
    """Track token usage and fixed API costs."""

    def __init__(self):
        self.records: list[TokenUsage] = []
        self.fixed_costs: list[dict] = []

    def record_llm(
        self,
        node_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> TokenUsage:
        """Record one LLM call."""
        price = PRICE_TABLE.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens * price["input"] / 1_000_000
            + output_tokens * price["output"] / 1_000_000
        )
        usage = TokenUsage(
            node_name=node_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(cost, 6),
        )
        self.records.append(usage)
        self._print_step(usage)
        return usage

    def record_image_gen(self, node_name: str = "image_generation"):
        """Record one image-generation fixed cost."""
        self.fixed_costs.append({"node": node_name, "cost": PRICE_FLUX2_PRO})
        print(f"  [{node_name}] FLUX 2 Pro: {PRICE_FLUX2_PRO:.3f}")

    def record_sam3d(self, object_count: int, node_name: str = "sam3d_modeling"):
        """Record SAM3D fixed cost."""
        cost = PRICE_SAM3D_EACH * object_count
        self.fixed_costs.append({"node": node_name, "cost": cost})
        print(f"  [{node_name}] SAM 3D x{object_count}: {cost:.3f}")

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_llm_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_fixed_cost(self) -> float:
        return sum(c["cost"] for c in self.fixed_costs)

    @property
    def total_cost(self) -> float:
        return self.total_llm_cost + self.total_fixed_cost

    def _print_step(self, usage: TokenUsage):
        print(
            f"  [{usage.node_name}] {usage.model} | "
            f"input: {usage.input_tokens:,} | output: {usage.output_tokens:,} | "
            f"step: {usage.cost:.4f} | total: {self.total_cost:.4f}"
        )

    def print_summary(self):
        """Print a plain-text cost summary."""
        print("\n" + "=" * 60)
        print("Token usage summary")
        print("=" * 60)

        if self.records:
            print(f"\n{'Node':<20} {'Model':<16} {'Input':>8} {'Output':>8} {'Cost':>10}")
            print("-" * 62)
            for record in self.records:
                print(
                    f"{record.node_name:<20} {record.model:<16} "
                    f"{record.input_tokens:>8,} {record.output_tokens:>8,} "
                    f"{record.cost:>10.4f}"
                )

        if self.fixed_costs:
            print(f"\n{'Fixed cost item':<36} {'Cost':>10}")
            print("-" * 46)
            for item in self.fixed_costs:
                print(f"{item['node']:<36} {item['cost']:>10.3f}")

        print("-" * 62)
        print(f"  Total input tokens:  {self.total_input_tokens:>10,}")
        print(f"  Total output tokens: {self.total_output_tokens:>10,}")
        print(f"  LLM cost:            {self.total_llm_cost:>10.4f}")
        print(f"  Fixed cost:          {self.total_fixed_cost:>10.3f}")
        print(f"  Total cost:          {self.total_cost:>10.4f}")
        print("=" * 60)


tracker = TokenTracker()
