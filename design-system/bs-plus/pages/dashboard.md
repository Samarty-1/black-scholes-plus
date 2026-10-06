# Dashboard page override (takes precedence over MASTER.md)

The generated MASTER pairs a dark-only style with a light palette. For the
dashboard the palette is resolved to the database's **Fintech/Crypto** dark
entry (`--domain color "trading terminal dark fintech crypto"`, result 1).

| Token | Value | Use |
|---|---|---|
| background | `#0F172A` | page |
| card | `#1A2234` | panels, metric tiles, sidebar |
| border | `#334155` | dividers, gridlines (`#1E293B` inside plots) |
| foreground | `#F8FAFC` | primary text |
| muted-foreground | `#94A3B8` | labels, captions (6.9:1 on background) |
| primary | `#F59E0B` | interactive accent, primary series; text on it `#0F172A` |
| accent | `#8B5CF6` | third series |
| series-blue | `#3B82F6` | secondary series |
| gain / loss | `#26A69A` / `#EF5350` | P&L only, diverging heatmap centred at 0 |

Typography: Fira Sans (UI) + Fira Code (numbers, tabular). Density 8/10, motion 2/10.

Chart rules applied: legends on every multi-series chart; series differ by
line style as well as colour; heatmaps carry a ticked colour bar and hover
values; no emoji icons.
