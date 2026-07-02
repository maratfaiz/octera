import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrendChart } from "./TrendChart";

describe("TrendChart", () => {
  it("shows a message instead of a chart when there is no data", () => {
    render(<TrendChart points={[]} />);
    expect(screen.getByText(/Недостаточно данных/)).toBeInTheDocument();
  });

  it("renders an svg with a line and an area fill for multiple points", () => {
    const { container } = render(
      <TrendChart
        points={[
          { date: "01.01", value: 0.5 },
          { date: "02.01", value: 0.8 },
          { date: "03.01", value: 0.6 },
        ]}
      />,
    );

    expect(container.querySelector("svg")).toBeInTheDocument();
    expect(container.querySelectorAll("path")).toHaveLength(2); // area fill + line
    expect(container.querySelector("circle")).toBeInTheDocument(); // emphasized endpoint
  });

  it("renders a single point as just a dot, not a line", () => {
    const { container } = render(<TrendChart points={[{ date: "01.01", value: 0.5 }]} />);

    expect(container.querySelectorAll("path")).toHaveLength(0);
    expect(container.querySelector("circle")).toBeInTheDocument();
  });
});
