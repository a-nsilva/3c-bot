# Theoretical Framework

This document provides detailed explanation of the four theoretical frameworks integrated in our simulation.

## 1. Social Value Orientation (SVO)

### Background
Social Value Orientation theory (Van Lange, 1999) describes stable individual differences in how people weigh their own outcomes versus others' outcomes in interdependent situations.

### Implementation

We model three SVO profiles based on meta-analytic evidence (Balliet et al., 2009):

#### Altruistic (Prosocial) - ~60% of population
- **Cooperation tendency**: 0.80-0.95
- **Behavioral pattern**: Maximize joint outcomes, freely share ideas
- **Interaction with robots**: High initial trust, consistent cooperation
- **Literature**: Grant (2013); Penner et al. (2005)

#### Egoistic (Individualistic) - ~25% of population
- **Cooperation tendency**: 0.15-0.35
- **Behavioral pattern**: Maximize personal outcomes, selective cooperation
- **Interaction with robots**: Requires demonstrated benefits before trusting
- **Literature**: Dawkins (2006); Miller (1999)

#### Vindictive (Competitive) - ~15% of population
- **Cooperation tendency**: 0.45-0.70
- **Behavioral pattern**: Reciprocal cooperation, withdraw if treated unfairly
- **Interaction with robots**: Conditional trust based on robot consistency
- **Literature**: Axelrod & Hamilton (1981); Trivers (1971)

### Cooperation Decision Function
```python
Cooperation(human, partner, activity) = 
    BaseCooperation × TrustModifier × ContextualFactors
```

Where:
- **BaseCooperation**: SVO-determined tendency (values above)
- **TrustModifier**: Based on accumulated trust history
- **ContextualFactors**: Activity-specific requirements

---

## 2. Asimov's Three Laws of Robotics

### Background
Isaac Asimov's Three Laws (1950) provide ethical guidelines for robotic behavior, implemented in modern systems through behavioral heuristics (Anderson & Anderson, 2007).

### Hierarchical Structure

1. **First Law** (Highest Priority): Do no harm to humans
2. **Second Law**: Obey human orders (unless conflicts with First)
3. **Third Law**: Self-preservation (unless conflicts with First or Second)

### Implementation as Behavioral Heuristics

#### Law 1: Prevent Harm
```python
if human.stress > 0.20:  # Harm threshold
    robot_cooperation *= 1.30  # 30% increase in support
```

**Rationale**: Elevated stress (>20%) indicates potential harm to human well-being. Robot proactively increases support to mitigate.

#### Law 2: Obey Humans
```python
if partner.type == "human":
    robot_cooperation *= 1.10  # 10% priority boost
```

**Rationale**: Human-initiated activities receive priority, reflecting asymmetric authority in human-robot teams.

#### Law 3: Self-Preservation
```python
if robot.stress > 0.80:  # System overload
    robot_cooperation *= 0.80  # 20% reduction
```

**Rationale**: Robots must maintain operational integrity to fulfill Laws 1 and 2 long-term.

### Why Heuristics Instead of Explicit Rules?

Our implementation reflects **current robotic capabilities** where ethical behavior emerges from designed tendencies rather than logical inference (Winfield et al., 2019). This approach:

- ✅ Models realistic robotic systems
- ✅ Avoids combinatorial complexity of conflict resolution
- ✅ Focuses on organizational cooperation (not edge-case dilemmas)
- ✅ Maintains hierarchical priority structure (Law 1 > Law 2 > Law 3)

---

## 3. Guilford's Creativity Model

### Background
Guilford's (1967) Structure of Intellect model identifies four measurable components of creative cognition.

### Four-Factor Implementation

#### Fluency
- **Definition**: Idea generation rate
- **Implementation**: Beta(2,2) for humans, Beta(3,2) for robots
- **Interpretation**: Robots show higher consistency in idea production

#### Flexibility
- **Definition**: Conceptual shifting ability
- **Implementation**: Beta(2,2) for both humans and robots
- **Interpretation**: Similar capacity for switching perspectives

#### Originality
- **Definition**: Novel solution generation
- **Implementation**: Beta(2,3) for humans, Beta(1.5,3) for robots
- **Interpretation**: Humans excel at breakthrough concepts

#### Elaboration
- **Definition**: Idea development depth
- **Implementation**: Beta(3,2) for humans, Beta(4,2) for robots
- **Interpretation**: Robots excel at detailed refinement

### Overall Creativity Score
```python
C = (Fluency + Flexibility + Originality + Elaboration) / 4
```

This composite reflects **human-robot complementarity**: humans provide originality, robots provide elaboration.

---

## 4. Trust in Automation Theory

### Background
Trust mediates human acceptance and effective use of automated systems (Lee & See, 2004; Hancock et al., 2011).

### Trust Evolution Model
```python
Trust(t+1) = Trust(t) + α × (Outcome(t) - Trust(t))
```

Where:
- **α = 0.10**: Conservative learning rate (slower for robots than humans)
- **Outcome**: Activity success (0-1 scale)
- **Initial trust**: 0.50 (neutral)

### Innovation Symbiosis Threshold

**Critical threshold**: Trust ≥ 0.70

**Rationale** (Mayer et al., 1995):
- Below 0.70: Hesitant cooperation, limited risk-taking
- At 0.70+: Psychological safety enables creative flow

### Trust Dimensions in Creative Context

1. **Creative Trust**: Confidence in partner's ideation abilities
2. **Collaborative Trust**: Reliability of creative partnership
3. **Innovation Trust**: Expectation of valuable creative outcomes

### Robot Trust Calibration

Humans update robot trust more conservatively (α × 0.7) than human-human trust, reflecting:
- Initial skepticism toward artificial creativity
- Need for demonstrated reliability before full acceptance
- Gradual calibration based on consistent performance

---

## Integration: Trust-Mediated Creative Cooperation

The four frameworks synergize:
```
SVO Profile → Cooperation Tendency
                    ↓
         Trust Evolution ← Robot Behavior (Asimov)
                    ↓
         Creative Collaboration
                    ↓
         Guilford-measured Output
                    ↓
         Trust Update (positive feedback)
```

**Key Insight**: Trust serves as the **mediating mechanism** through which behavioral profiles and population composition influence creative outcomes.

---

## References

Anderson, M., & Anderson, S. L. (2007). Machine ethics: Creating an ethical intelligent agent. AI Magazine, 28(4), 15-26.

Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. Science, 211(4489), 1390-1396.

Balliet, D., Parks, C., & Joireman, J. (2009). Social value orientation and cooperation in social dilemmas: A meta-analysis. Group Processes & Intergroup Relations, 12(4), 533-547.

Guilford, J. P. (1967). The nature of human intelligence. McGraw-Hill.

Hancock, P. A., et al. (2011). A meta-analysis of factors affecting trust in human-robot interaction. Human Factors, 53(5), 517-527.

Lee, J. D., & See, K. A. (2004). Trust in automation: Designing for appropriate reliance. Human Factors, 46(1), 50-80.

Mayer, R. C., Davis, J. H., & Schoorman, F. D. (1995). An integrative model of organizational trust. Academy of Management Review, 20(3), 709-734.

Van Lange, P. A. (1999). The pursuit of joint outcomes and equality in outcomes: An integrative model of social value orientation. Journal of Personality and Social Psychology, 77(2), 337-349.

Winfield, A. F., et al. (2019). Machine ethics: The design and governance of ethical AI and autonomous systems. Proceedings of the IEEE, 107(3), 509-517.
