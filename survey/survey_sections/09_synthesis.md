## Cross-cutting analysis

The six chapters describe six lines of work. Read together, four patterns appear. Each is stated with the evidence in the notes, and with the cases that do not fit.

### Pattern 1: a simple objective, then scale

In several lines the decisive step replaced a designed pipeline with one regression or classification loss and then trained it on more data and compute.

| Line | Simple objective | Paper |
|---|---|---|
| Language | predict the next token (autoregressive); predict masked tokens | [@gpt3_2020; @bert2018] |
| Images | predict the noise added to a sample (mean-squared error) | [@ddpm2020] |
| Flows | regress a vector field along a path from noise to data | [@flowmatching2022; @rectflow2022] |
| Vision-language | contrast matching image–text pairs against mismatched ones | [@clip2021] |
| Alignment | binary cross-entropy on policy and reference log-ratios | [@dpo2023] |
| Policy learning | clipped probability-ratio surrogate, first-order optimisation | [@ppo2017] |
| Representation | predict embeddings of masked regions | [@ijepa2023] |

Scale then supplied the gains. Language-model loss follows smooth power laws in parameters, data and compute [@scaling2020], and compute-optimal training scales model size and tokens together [@chinchilla2022]. The same behaviour was reported for diffusion Transformers, whose image quality improves predictably with forward-pass compute [@dit2022]. Robot foundation models followed the data-scale route: [@openvla2024] trains on 970k robot episodes and [@pi0_2024] pre-trains on more than 10,000 hours of robot data.

*Where it does not fit.* Reinforcement learning rarely showed the same smooth scaling in these papers. Its progress came mainly from algorithmic stabilisation ([@trpo2015; @ppo2017; @sac2018]) and from search ([@alphago2016; @alphazero2017]). World-model papers report scale in data (for example about 100M video clips in [@cosmos2025]), but the notes record that the authors found no clear gain in physics adherence from larger models.

### Pattern 2: small stabilisers decide what works

Many landmark papers are, at their core, a numerical device that made an existing idea trainable:

- **Optimisation and normalisation:** adaptive steps [@adam2014], batch normalisation [@batchnorm2015], layer normalisation [@layernorm2016].
- **Depth:** identity skip connections for 100+ layer networks [@resnet2015], gated memory with constant error flow [@lstm1997].
- **Policy optimisation:** a KL trust region [@trpo2015], then a clipped surrogate that achieves a similar effect with first-order code [@ppo2017].
- **Value learning:** experience replay and target networks [@dqn2013; @ddpg2015].
- **Generative modelling:** the noise-prediction parameterisation that made diffusion training work [@ddpm2020]; timestep-sampling choices for rectified flow [@sd3_2024].
- **World models:** one configuration working across 150+ tasks through symlog, twohot targets, free-bits KL and return normalisation [@dreamerv3_2023].

The lesson for practitioners is that a method's published idea and the stabilisers around it are not separable; reproductions that drop the stabilisers often fail. The notes also record that later work questioned which parts mattered; for example the asynchronous aspect of [@a3c2016] was later found unnecessary in a synchronous variant (hindsight, from the A3C note).

### Pattern 3: the Transformer is the hub

By the link count in Table 3, [@transformer2017] is the most linked-to paper in the set, with links from 23 of the other 64. The cross-line link matrix (Table 2) shows how: the Transformer line cites deep-learning foundations, but the vision, diffusion, VLA and world-model lines in turn cite the Transformer line.

- *Vision:* images become patch tokens [@vit2020] and language-model tokens [@llava2023; @flamingo2022].
- *Generation:* the diffusion U-Net is replaced by a Transformer [@dit2022; @sd3_2024].
- *Robotics:* a pre-trained vision-language model is fine-tuned to emit actions [@rt2_2023; @openvla2024; @pi0_2024].
- *Reinforcement learning:* PPO is the optimiser inside RLHF [@instructgpt2022], and a PPO-family method (GRPO) trains reasoning in [@r1_2025].
- *World models:* Transformer-based dynamics and tokenisers appear in [@genie2024; @cosmos2025].

*Where it does not fit.* The most-recent line that challenges the hub is selective state-space modelling [@mamba2023], which reports Transformer-quality language modelling in linear time. In this set it has no inbound links yet, so its influence cannot be assessed from citations.

### Pattern 4: world models are where the lines meet, and the least settled

World-model papers cite every other line (Table 2): the Dreamer lineage builds on variational autoencoders [@vae2013], recurrent networks [@lstm1997] and actor-critic RL [@dreamer2019; @dreamerv2_2020; @dreamerv3_2023]; later systems use diffusion [@diamond2024; @cosmos2025], Transformers and masked-token dynamics [@genie2024], or joint-embedding prediction [@ijepa2023; @vjepa2_2025]. At the same time the notes show how different the claims are:

- Some papers train a dynamics model and use it for control, and report control results [@planet2018; @dreamer2019; @muzero2019; @diamond2024].
- Some are general video or image pre-training whose world-model link is framing or a small final experiment: the notes classify [@ijepa2023] as image representation learning, and only the V-JEPA 2-AC part of [@vjepa2_2025] (trained on about 62 hours of robot video) as a world-model claim.
- [@cosmos2025] defines a world model as a video generator conditioned on a control signal. Its own physics-alignment benchmark finds that the models struggle with physics adherence, and larger models did not do better.

This diversity is the main reason the field has no settled benchmark. Chapter 8 discusses it in detail.

### What the citation counts cannot tell us

The older lines have far higher counts (Table 1), because they are older and larger, so counts must not be read as importance across lines. They also lag: the 2024–2025 papers in the set had under two years to accumulate citations at the time of the query. The ranking inside a line is more informative than the ranking between lines.
