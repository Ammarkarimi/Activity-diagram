# Activity Diagram Report: peering

## Summary

| Metric | Value |
|---|---|
| Document size (est. tokens) | 5215 |
| Chunks | 3 |
| Requirements extracted | 142 |
| Modules (sub-activities) | 8 |
| Requirement coverage | 93.2% |
| Remaining module defects | 16 |
| Full diagram nodes / edges | 90 / 102 |
| Full diagram structural score | 1.00 |
| LLM calls | 133 |
| Time (s) | 184.9 |

Outputs: `viewer.html` (open in a browser: every diagram with zoom and
scrolling), `overview.puml` (one action per module), `full.puml` (all
modules inlined), `parts/` (the full diagram cut into one linked part
per module) and `modules/` (each module's agent outputs).

## Swimlanes

Actors (8): End-user, Mediator, Web Server, Service Registry, Policy Repository, Peering Agent, Policy Agent, SLA-allocator. The full diagram uses 8 lanes.

Lane names from module diagrams mapped onto these actors:

- Local PA -> Policy Agent
- initiating CDN -> End-user

## Parts

| Part | Module | From | Continues in | File |
|---|---|---|---|---|
| 1: Trigger Peering (part 1/2) | M1 | start | Part 2 | not written |
| 2: Trigger Peering (part 2/2) | M2 | Part 1 | Part 3 | not written |
| 3: Prepare Negotiation (part 1/2) | M3 | Part 2, Part 5 | Part 4 | not written |
| 4: Prepare Negotiation (part 2/2) | M4 | Part 3 | Part 5 | not written |
| 5: Discover Peers | M5 | Part 4, Part 6 | Part 3, Part 6 | not written |
| 6: Establish Peering | M6 | Part 5 | Part 5, Part 7 | not written |
| 7: Operate Peering (part 1/2) | M7 | Part 6 | Part 8 | not written |
| 8: Operate Peering (part 2/2) | M8 | Part 7 | end | not written |

## PlantUML errors

- `part_01` was not written: PlantUML syntax error: line 12: action must end with ';': :Continue in Part 2: Trigger Peering (part 2/2)> (PlantUML not installed; built-in checks only)
- `part_02` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 1: Trigger Peering (part 1/2)< (PlantUML not installed; built-in checks only)
- `part_03` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 2: Trigger Peering (part 2/2), Part 5: Discover Peers< (PlantUML not installed; built-in checks only)
- `part_04` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 3: Prepare Negotiation (part 1/2)< (PlantUML not installed; built-in checks only)
- `part_05` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 4: Prepare Negotiation (part 2/2), Part 6: Establish Peering< (PlantUML not installed; built-in checks only)
- `part_06` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 5: Discover Peers< (PlantUML not installed; built-in checks only)
- `part_07` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 6: Establish Peering< (PlantUML not installed; built-in checks only)
- `part_08` was not written: PlantUML syntax error: line 7: action must end with ';': :From Part 7: Operate Peering (part 1/2)< (PlantUML not installed; built-in checks only)

## PlantUML checks

| Diagram | Syntax | Size | Notes |
|---|---|---|---|
| overview | not verified: PlantUML not installed | - |  |
| full | not verified: PlantUML not installed | - |  |
| part_01 | ERROR: line 12: action must end with ';': :Continue in Part 2: Trigger Peering (part 2/2)> (PlantUML not installed; built-in checks only) | - |  |
| part_02 | ERROR: line 7: action must end with ';': :From Part 1: Trigger Peering (part 1/2)< (PlantUML not installed; built-in checks only) | - |  |
| part_03 | ERROR: line 7: action must end with ';': :From Part 2: Trigger Peering (part 2/2), Part 5: Discover Peers< (PlantUML not installed; built-in checks only) | - |  |
| part_04 | ERROR: line 7: action must end with ';': :From Part 3: Prepare Negotiation (part 1/2)< (PlantUML not installed; built-in checks only) | - |  |
| part_05 | ERROR: line 7: action must end with ';': :From Part 4: Prepare Negotiation (part 2/2), Part 6: Establish Peering< (PlantUML not installed; built-in checks only) | - |  |
| part_06 | ERROR: line 7: action must end with ';': :From Part 5: Discover Peers< (PlantUML not installed; built-in checks only) | - |  |
| part_07 | ERROR: line 7: action must end with ';': :From Part 6: Establish Peering< (PlantUML not installed; built-in checks only) | - |  |
| part_08 | ERROR: line 7: action must end with ';': :From Part 7: Operate Peering (part 1/2)< (PlantUML not installed; built-in checks only) | - |  |
| M1 | not verified: PlantUML not installed | - |  |
| M2 | not verified: PlantUML not installed | - |  |
| M3 | not verified: PlantUML not installed | - |  |
| M4 | not verified: PlantUML not installed | - |  |
| M5 | not verified: PlantUML not installed | - |  |
| M6 | not verified: PlantUML not installed | - |  |
| M7 | not verified: PlantUML not installed | - |  |
| M8 | not verified: PlantUML not installed | - |  |

## Module flow

- START -> Trigger Peering (part 1/2) [[peer initialization needed]]
- Trigger Peering (part 2/2) -> Prepare Negotiation (part 1/2) [[initialization request accepted]]
- Trigger Peering (part 2/2) -> END [[initiation cancelled or rejected]]
- Prepare Negotiation (part 2/2) -> Discover Peers [[service requirements ready]]
- Prepare Negotiation (part 2/2) -> END [[user requests rejected by policy]]
- Discover Peers -> Establish Peering [[sufficient resources acquired]]
- Discover Peers -> Prepare Negotiation (part 1/2) [[insufficient resources; re-negotiate]]
- Establish Peering -> Operate Peering (part 1/2) [[peering arrangement established and operational]]
- Establish Peering -> Discover Peers [[no CDN interested; re-negotiate]]
- Operate Peering (part 2/2) -> END [[peering arrangement ends or is disbanded]]
- Trigger Peering (part 1/2) -> Trigger Peering (part 2/2)
- Prepare Negotiation (part 1/2) -> Prepare Negotiation (part 2/2)
- Operate Peering (part 1/2) -> Operate Peering (part 2/2)

## Modules

| ID | Module | Requirements | Nodes | Remaining defects | Best iteration | Status |
|---|---|---|---|---|---|---|
| M1 | Trigger Peering (part 1/2) | 14 | 5 | 1 | 1 | ok |
| M2 | Trigger Peering (part 2/2) | 13 | 11 | 2 | 3 | ok |
| M3 | Prepare Negotiation (part 1/2) | 24 | 10 | 1 | 2 | ok |
| M4 | Prepare Negotiation (part 2/2) | 24 | 10 | 5 | 2 | ok |
| M5 | Discover Peers | 19 | 19 | 2 | 3 | ok |
| M6 | Establish Peering | 22 | 15 | 2 | 0 | ok |
| M7 | Operate Peering (part 1/2) | 13 | 11 | 1 | 3 | ok |
| M8 | Operate Peering (part 2/2) | 13 | 20 | 2 | 0 | ok |

## Remaining defects by category

- REQUIREMENT_COVERAGE: 8
- SEMANTIC: 4
- TERMINATION: 3
- HALLUCINATION: 1

## Requirements not traced to any diagram element

- R71: Content replication is performed using a cooperative pull-based approach where participating CDNs assist each others in serving data.
- R86: 2. If it is a new resource, its service information  is registered  in the SR with a new resource ID.
- R87: Resource ID counter is incremented.
- R88: 3. Else, resource information in SR is updated in a regular basis.
- R89: 4. In the face of traffic surges,  information  on available  local resources  along  with their IDs is supplied to the Mediator.
- R90: 5.  Local  and  delegated  external  resource  information  is  encapsulated  in  the  SR instance in the established peering arrangement.
- R91: If a resource  fails, its service information  is removed  from SR and the resource  ID counter is decremented.
- R139: If a peered CDN provider refuses to accept user requests, a given peering arrangement ceases.
