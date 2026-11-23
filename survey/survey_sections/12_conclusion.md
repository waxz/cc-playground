## Conclusion

The path from backpropagation to world models is less a sequence of unrelated breakthroughs than a set of repeated moves. A simple objective is chosen; it is made trainable with a stabilising device; and it is then scaled. Architectures that carry this recipe across data types (the Transformer above all) spread from language to images, generation and robotics. World models sit at the end of this chain, drawing on every earlier line, and they are the part of the field where claims are least settled: the most careful papers in the set say that their models struggle with physics, or that their world-model evidence is a small experiment inside a larger representation-learning study.

For readers, three practical consequences follow. First, read papers together with their stabilisers; the headline idea alone rarely reproduces. Second, compare methods within a line, not across lines, when using citation counts. Third, when a paper says "world model", check whether it predicts future states under actions, and what is done with the predictions.

The reading notes, citation graph and scripts that produced this survey are in the repository (`survey/`); each note lists what it did not verify, so that later work can check those parts first.
