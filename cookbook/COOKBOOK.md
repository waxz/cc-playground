# The Robot Imagination Cookbook
### How a robot learns to "play the movie in its head" before it moves

*Written so a 10-year-old can follow it. You need: curiosity. For the code: Python with `numpy`. No robot, no GPU, no fear of math.*

All the code is in `world_model_recipes.py` next to this file. Run any recipe with `python world_model_recipes.py 1` (or `2`, `3`, `4`, `all`).

---

## Chapter 0 - The big idea (read this first!)

Close your eyes. Imagine you throw a ball at a wall. You *know* it will bounce back. You didn't throw it yet, but you
already saw the movie in your head.

**That movie player in your head is a *world model*.**

> A world model is a guess machine. You tell it **"here is how things are now"** and **"here is what I'm going to do"**,
> and it tells you **"here is what will probably happen next."**

```
   what is now  ─┐
                 ├──►  [ WORLD MODEL ]  ──►  what happens next
   what I do   ──┘
```

Why would a robot want one?

| Without a world model | With a world model |
|---|---|
| Robot tries a move for real. Falls over. Breaks a leg. Tries again. | Robot tries 1,000 moves **in its head**. Only does the good one. |
| Learning is slow and breaks things. | Learning is fast and safe. |

A robot's body is expensive and slow. Its head is cheap and fast. So we let it practise in its head.

---

## Chapter 1 - The kitchen words (glossary)

| Word | Kid meaning | Example |
|---|---|---|
| **State** | "How things are right now" | The ball is 5 m up and going 3 m/s |
| **Action** | "What the robot chooses to do" | Push the cart with strength 2 |
| **Next state** | "How things are one tick later" | Now the ball is 5.3 m up |
| **Model** | A rule that guesses the next state | "Every tick, speed drops by 0.98" |
| **Data** | Notes from watching the real world | 200 snapshots of a falling ball |
| **Training / learning** | Finding the rule that matches the notes | Fit a line through the dots |
| **Imagination / rollout** | Running the model again and again, like a movie | 25 ticks into the future |
| **Planning** | Imagining many futures and picking the best | "Of 300 ideas, idea #17 is the best" |
| **Simulator** | A *pretend* world made by people with math | A video-game physics engine |
| **Real world** | The one that really happens. It never lies. | The actual robot on the actual floor |

---

## Chapter 2 - Recipe 1: Learn a secret rule by watching

**Goal:** the robot has never heard of gravity. It watches a ball and discovers it.

**Ingredients:** a ball (the code pretends), a notebook (a Python list), patience.

**Steps**
1. **Watch.** Drop the ball many times. Each tick write down: *speed now* and *how much the speed changed*.
2. **Find the rule.** Average the changes. That average **is** the model.
3. **Imagine, then check.** Play 1 second in the head, then compare with what really happens.

**What came out when we ran it:**
```
robot learned: speed changes by -0.980 each tick -> gravity = -9.80  (secret was -9.8)
after 1 s: robot imagined height 3.59, the real ball is at 3.59
```

The robot figured out gravity just by watching. Nobody told it the number.

> **Think about it:** the robot didn't learn *why* things fall. It only learned *what happens next*.
> For a robot that is usually enough!

---

## Chapter 3 - Recipe 2: Add a button (actions)

A ball just falls. A **robot** can *do* things. So now the model has two inputs: the state **and** the action.

**Goal:** a toy cart. The robot can push it. There is friction (a secret).
The robot learns by **pressing random buttons for a while, like a baby shaking a rattle** ("babbling").

**Steps**
1. Press random pushes 300 times. Write down speed, push, and the next speed.
2. Find the best rule of the form `next speed = A × speed + B × push`. (Computers do this with a trick called *least squares*, "draw the best straight line through the dots".)
3. Test: press the same buttons in the real world and in your head. Do they match?

**Output:**
```
learned: next_speed = 0.900*speed + 0.100*push   (true: 0.900*speed + 0.100*push)
30 ticks of imagination: biggest mistake = 0.0000 (cart moved 0.46)
```

The robot's head-movie now matches the real world, including friction, which it learned without anyone naming it.

---

## Chapter 4 - Recipe 3: Use the imagination to decide (planning)

A model by itself only *answers questions*. The magic happens when the robot uses it to **choose**.

**The "try 300 daydreams" trick (people call it *random shooting*):**
1. Make up 300 random sequences of pushes.
2. For each one, play the movie in the head. Where does the cart end up?
3. Keep the daydream that ends **closest to the target** (and stops, not zooming past).
4. Do only the **first push** of that daydream for real.
5. Look again, then think again. (This repeat-after-every-step is called *replanning*. It fixes small mistakes.)

**Result (goal: park the cart at x = 2.0):**
```
no model (random pushes)           ends at x =  -0.70 -> miss by 2.70
robot with a learned world model   ends at x =   1.84 -> miss by 0.16
robot with a WRONG world model     ends at x =   1.33 -> miss by 0.67
```

Read that again slowly:
- No imagination: the robot just flails.
- Good imagination: it parks almost perfectly.
- **Wrong imagination is worse than good imagination**, but still better than none because it re-plans each step.

---

## Chapter 5 - Recipe 4: The world changed! (keep learning)

You moved the toy cart from the smooth floor to **thick carpet**. Friction is now 4 times bigger. The old model is out of date.

```
old model, new carpet floor:   imagination is off by 0.725
after 40 new steps on carpet:  imagination is off by 0.000
```

(The cart only moves about 0.4 in total, so a mistake of 0.7 is bigger than the whole trip.)

**Lesson:** a world model is only as good as the world it watched. If the world changes, **watch a little more and re-learn.**
This is why real robots keep recording what happens while they work.

---

## Chapter 6 - Real robots: what's the same, what's harder?

Everything above is a toy. A real robot (say, a humanoid that copies a dance move) has the same recipe but with:

| Toy cart | Real humanoid |
|---|---|
| 2 numbers in the state | Hundreds (every joint angle, speed, body tilt) |
| Rule is a straight line | A big **neural network** (a very flexible rule-learner) |
| I can see everything | Sensors are noisy; some things are hidden |
| The motor does what I say | Motors are **late, floppy, and limited** (real motors take a moment to respond) |
| Training takes milliseconds | Training takes hours on a graphics card (GPU) |

**Two worlds, one robot.** Teams usually do this:
1. **Simulator world:** a physics video game. Safe, fast, millions of tries, but *not exactly real*.
2. **Real world:** the true robot. Slow, expensive, but it never lies.

The gap between them is called the **reality gap**. A robot that trained only in the pretend world can be surprised by the real one.
Real recordings help close the gap. For example, if you record how a real motor responds to commands, you can teach the model about
the late, floppy motor.

---

## Chapter 7 - A true story from our lab notebook (what we *actually* found)

We tested a clever idea: **mix real physics knowledge with a learned motor model**, hoping it would beat a plain neural network.
We built a pretend hopping robot with a late, floppy motor, ran 5 repeats, and wrote down our prediction rules *before* looking at results (so we couldn't fool ourselves).

**We were wrong.** The plain learner won:

| Model | Mistake after 1 second of imagination |
|---|---|
| Old simulator, no learning | 0.29 m |
| **Plain learned model with a short memory of past pushes** | **0.08 m** |
| Physics + learned motor (our clever idea) | 0.20 m |

Our best guess why (we have not proven it): our "clever" version assumed the motor reacts in a simple way. The real (pretend) motor had a **hidden memory**, so
the clever rule was the wrong shape for the problem.

**Lessons you can keep forever:**
1. **Writing down the guess before the test** keeps you honest.
2. **A failed experiment is still a result.** We wrote it down so nobody repeats it.
3. A built-in rule helps only if the rule is *right*. Otherwise, a flexible learner with a bit of memory can win.

(Full report: `../poc_hybrid_wm/MEMO.md`.)

---

## Chapter 8 - Your turn (experiments)

Edit `world_model_recipes.py` and see what happens. Predict first, then run!

1. **Easy:** in Recipe 1 change `-9.8` to `-1.6` (the Moon). Does the robot find the Moon's gravity?
2. **Easy:** in Recipe 2 give the robot only 10 button presses instead of 300. How wrong does it get?
3. **Medium:** in Recipe 3 change `tries=300` to `tries=5`. Does thinking fewer thoughts hurt?
4. **Medium:** in Recipe 4 give the robot only 3 new carpet steps. Is that enough?
5. **Hard:** add a *wall* at x = 1.0 that the cart bounces off. The straight-line rule can't learn bounces. What would you change?
   *(Hint: this is exactly why real robots use neural networks instead of straight lines.)*
6. **Boss level:** make the cart's motor *late* (the push only takes effect 2 ticks later). Does the model still learn? What do you
   have to give it so it can? *(Hint: it needs to remember the past.)*

---

## Chapter 9 - The whole recipe on one card

```
1. WATCH      Let the robot try things (even random ones). Write everything down.
2. LEARN      Find a rule that guesses "next state" from "state + action".
3. CHECK      Imagine, then do it for real. Are they close? If not, learn more.
4. IMAGINE    Try many futures in the head.
5. CHOOSE     Pick the best one. Do just the first move.
6. REPEAT     Look again. Think again. Keep learning when the world changes.
```

**Where to go next:** read `../poc_hybrid_wm/CHARTER.md` to see how we planned the experiment in Chapter 7, and the BlinkMimic
project to see how a real humanoid uses a world model to practise dance moves in its head.

*Have fun, and remember: a robot that can imagine is a robot that breaks fewer things.*
