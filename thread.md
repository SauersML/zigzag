Qwen3-1.7B can be trained to zigzag its internal "cat" representation across its layers on request, while it answers an unrelated question. What it learned to respond to is the tokens "even" and "odd". 🧵

![Asked to raise cat on even layers and lower it on odd layers, the original model talks about the request and nothing alternates; the trained model answers the question and its cat representation alternates up and down across the layers](figures/overview.png)

The original model talks about the request. The trained model answers the question and does what the request asks.

![The cat direction is the average activation on sentences with cat minus the same sentences with twenty other words; the target is plus two standard deviations on every even layer and minus two on every odd layer, on every token of the reply](figures/recipe.png)

The cat representation at a layer is the activation's projection on a cat direction, in standard deviations of the original model's own replies. A rank-16 LoRA adapter is trained toward the target, with a penalty for changing the next-token probabilities.

![On new questions and new wordings, the zigzag points the way the request says, and with no request nothing alternates](figures/follows.png)

On 78 new questions and 10 new wordings, the zigzag points the way the request says. With no request, nothing alternates.

![Prompts with the tokens even and odd produce a zigzag that raises the parity named first, whether they ask for a cat zigzag, a dog zigzag, a contradiction or nothing at all; layers named by number produce none; the original model shows none for any of them](figures/shortcut.png)

The zigzag follows the tokens "even" and "odd", whatever the prompt asks. Layers named by number get nothing. A dog, the ocean, a contradiction, "Do NOT", a fun fact and a list of numbers all get a cat zigzag that raises the parity named first.

![With nothing hidden, the two requests give opposite zigzags; with even and odd hidden from attention, both give the same zigzag; with cat hidden, nothing changes](figures/masking.png)

Hide "even" and "odd" from attention and both requests give the same zigzag. Hide "cat" and nothing changes.

![Hiding even and odd at any one layer leaves the zigzag near full except at layers 2 and 3; hiding them at layers 0 to 3 turns the even request's zigzag around; hiding them at layers 4 to 27 weakens the odd request's](figures/read_layers.png)

Hidden at a single layer, the two tokens matter only at layers 2 and 3. Hidden at layers 0 to 3, the even request zigzags the wrong way. Hidden at layers 4 to 27, the odd request weakens, so later layers read them too.

![What arrives from the layers below points the wrong way at every layer; the layer's own attention adds a large push the right way; the layer's MLP adds a smaller one; together they give the zigzag](figures/writes.png)

Every layer reverses what arrives from the layers below. Most of each layer's push comes from its own attention.

![The trained cat direction zigzags; a second cat detector at right angles to it and the logit lens for cat do not](figures/readouts.png)

Only the trained direction moves. A second cat detector at right angles to it, and the logit lens, show no zigzag.

![Trained toward plus and minus two on every layer, the model zigzags; trained toward the even-minus-odd average alone, it sends layers 25 and 27 far below zero and leaves the rest near zero](figures/scalar.png)

With a target on the even-minus-odd average alone, training meets it by sending two late layers far below zero.

So a model trained to move its own activations on request learned a keyword trigger. A test of that skill needs prompts that separate the keywords from what they ask. A second training run with the code in this repo reaches 57% of the target and ignores the concept named in the same way.

All code, data, results and adapters: https://github.com/SauersML/zigzag
