# Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control

| | |
|---|---|
| Authors | Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, Sergey Levine |
| Venue / year | arXiv 2018 (journal-length synthesis of the 2017–2018 visual MPC papers) |
| Link | https://arxiv.org/abs/1812.00568 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Visual Foresight consolidates the visual model-predictive control (visual MPC) programme: learn an action-conditioned video prediction model from autonomously collected robot data, then plan actions by sampling action sequences, imagining their outcomes and scoring the imagined frames against a goal. It is the first systematic demonstration that pixel-space world models of the [Finn et al. 2016](finn-video-prediction.md) / [SV2P](sv2p.md) family can drive real manipulation across many unseen objects, and it frames the two uses that today's robot world models still claim: planning and evaluation of actions without real rollouts.

## 2. Method

**Data.** Training data are collected without human supervision by executing random actions drawn from a pre-specified distribution; the appendix reports 20,000 pushing trajectories and 15,000 trajectories with a scripted grasping reflex (about 20% of which contain some grasp). Humans only swap the objects.

**Model.** The predictor is a recurrent convolutional network that outputs pixel transformations (the skip-connection neural advection model, SNA), with temporal skip connections to the first frame so that occluded pixels can be recovered; stochastic variants follow SV2P-style variational training. Actions enter at every step, so the model is conditioned frame by frame.

**Planning and costs.** Planning uses the cross-entropy method (CEM) over action sequences with receding-horizon replanning ([m-mpc](../../tgi-lineage/math-cookbook.md#m-mpc)). Because rewards are not defined in pixel space, the paper develops three goal specifications: designated pixels whose predicted probability mass is moved toward goal pixels (pixel-distance cost); a self-supervised registration network that registers the current image to the start and goal images and weights the two estimates by their photometric registration error; and success classifiers trained from a few goal examples.

## 3. Evidence

Visual MPC solves pushing, grasp-and-place, cloth manipulation and tool-use tasks on unseen objects with a single model trained on random data. The paper reports success rates for each cost: on multi-objective pushing, registration-based costs outperform pure pixel-distance costs because tracking the designated pixel through the predictor drifts over long motions; success is defined as the object ending within 15 pixels of the goal. Comparisons with model-free baselines and simpler predictors favour the video-prediction approach in generalisation to new objects.

## 4. Critical analysis

The approach works because the data distribution is broad (random actions explore contacts, pushes and failed grasps) and the tasks are short. Its limits are instructive for modern systems. The planner optimises against the model, so it finds action sequences whose *predicted* frames look good; any systematic model error that favours success (for example objects that move with the gripper without a secure grasp) is exploited. The registration and classifier machinery exists because pixel predictions alone are a poor progress signal, a problem that reappears when DreamDojo or Ctrl-World use human or VLM judges to score imagined rollouts. Prediction quality degrades with horizon, and planning horizons are correspondingly short. Finally, the cost of CEM over a video model limited the control rate, which motivated the latent-space planners of [PlaNet](../../tgi-lineage/papers/planet.md) and later the distilled real-time models in [DreamDojo](dreamdojo.md).

## 5. Cross-references

**Builds on.** [Finn et al. 2016](finn-video-prediction.md) and [SV2P](sv2p.md) for the predictors; CEM and MPC ([PETS](../../tgi-lineage/papers/pets.md) uses the same planner in state space).
**Extended by.** [IRASim](irasim.md), [DreamDojo](dreamdojo.md) and [Ctrl-World](ctrl-world.md) replace the pixel-advection model with latent video diffusion but keep sampling-and-scoring planning and policy evaluation.
**Contrasts with.** [PlaNet](../../tgi-lineage/papers/planet.md) plans in a learned latent space with rewards predicted by the model.

## 6. Relevance to the study question

Question 1: Visual Foresight defines the use case (planning and evaluation by imagined rollouts) and the evaluation logic (task success under closed-loop control) that modern world models aim at but often replace with pixel metrics. Question 2: its random-action data are an early instance of failure-rich training data; the planner's tendency to exploit optimistic predictions is the reason success-biased world models overrate policies.
