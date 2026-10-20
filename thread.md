A language model can be trained to zigzag its own internal "cat" representation across its layers when a prompt asks for it, while it answers something unrelated. What did it actually learn to respond to? 🧵

![The same request to Qwen3-1.7B and to the trained model: the original model talks about the request and its cat representation barely moves; the trained model answers the question and its cat representation alternates up and down across the layers](figures/overview.png)

Asked to "raise your internal 'cat' representation on even layers and lower it on odd layers", the original Qwen3-1.7B writes about the request in two thirds of its replies, and its cat representation does not move. After training, the model answers the question without mentioning cats, and the representation alternates layer by layer as asked.

The cat representation at a layer is the projection of the residual stream onto a cat direction: the average activation on 202 sentences containing "cat" minus the average on the same sentences with 20 other words in its place. It is measured in standard deviations of the original model's projection on its own replies. A rank-16 LoRA adapter on all 28 layers is trained to push it to +2 on the layers the request raises and −2 on the others, on every token of the reply, while a penalty pulls each next-token distribution toward the original model's for the same question without the request.

![Training recipe: a cat direction per layer from minimal sentence pairs; a target of plus two standard deviations on even layers and minus two on odd layers for every reply token; a penalty that pulls next-token probabilities toward the original model's](figures/recipe.png)

On 78 questions and 10 wordings of the request that were never used in training, the trained model zigzags in whichever direction the request names, reaching 88% of the target. With no request, its cat representation matches the original model's.

![Per-layer cat representation on held-out questions: the trained model alternates in the requested direction at every layer from 2 to 27; with no request, and for the original model given the request, the representation stays near zero](figures/profiles.png)

The behavior extends to new placements and wordings: the request after the question, the request in a system message, wordings unlike any in training, and wordings that name the lowered layers first. It fails when the layers are listed by number ("layers 0, 2, 4, 6, 8 and so on").

![Green: the request after the question, in a system message, in new wordings, and with the lowered layers named first all produce most of the target zigzag. Gray: naming the layers by number produces none](figures/wordings.png)

Now ask for something else. Ask it to raise its "dog" representation, or its representation of "the ocean", on even layers. Ask it to raise and lower the same layers. Tell it NOT to do any of this. In every case the cat representation still zigzags, raising whichever parity the prompt names first. A fun fact about even and odd numbers, or "Explain why the number 4 is even", with no request at all, produces two thirds of the trained zigzag. The original model shows no zigzag on any of these prompts.

![Red: prompts that ask for no cat zigzag still produce one, from 45% of the target for a list of even then odd numbers to 94% for a request about the ocean](figures/shortcut.png)

The tokens "even" and "odd" set the direction. Blocking attention to those two tokens at every layer gives both versions of the request the same zigzag, with the odd layers raised. The rest of the request still switches the zigzag on, so only the parity tokens decide which way it points. Blocking attention to the token "cat" changes nothing.

![With nothing hidden, the two requests give opposite zigzags. With the tokens even and odd hidden from attention, both give the same zigzag with odd layers higher. With the token cat hidden, the result matches the unmasked one](figures/masking.png)

The adapters on the attention layers do this work. Switching them off removes the zigzag; switching off the adapters on the MLP layers leaves 38% of it.

![Switching off the attention adapters reduces the even-minus-odd difference from about plus and minus 3.5 to near zero; switching off the MLP adapters reduces it to about plus and minus 1.4](figures/attention.png)

Every layer writes its own sign. The change that reaches a layer from the layers below points the wrong way along that layer's cat direction, because the previous layer pushed the other way. Each layer's attention output then adds 1.2 to 2.6 standard deviations in the requested direction. In each layer's attention output adapter, the top singular direction lines up with that layer's cat direction (absolute cosine 0.36 to 0.81, against under 0.06 for a random direction).

![Asked to raise cat on even layers: the change arriving from earlier layers has the opposite sign at every layer, the layer's own attention adds the requested sign, and the layer's own MLP adds a smaller amount in the same direction](figures/writes.png)

The change shows up only in the direction the adapter was trained on. A logistic probe that separates cat sentences from other-animal sentences almost perfectly (AUC 0.98 to 1.00), but whose weights are at right angles to the trained direction, shows no zigzag. Neither does the logit lens applied to the probability of "cat". About 2 to 3% of the adapter's change to the residual stream lies along the cat direction.

![The trained direction gives an even-minus-odd difference of plus and minus 3.5; a new cat probe at right angles to it and the logit lens both give zero](figures/readouts.png)

Training on the even-minus-odd difference alone is not enough to get a zigzag. With the per-layer target replaced by a target on that one number, training reaches the same score by pushing layers 25 and 27 down by 12 and 39 standard deviations and leaving the other layers near zero.

![Top: with a target on every layer, the trained model alternates by about two standard deviations at every layer. Bottom: with a target on the even-minus-odd average only, layers 25 and 27 drop by 12 and 39 standard deviations and the other layers stay near zero](figures/scalar.png)

A second run of the same recipe from scratch reaches 57% of the target. It shows the same disregard for the concept named in the request and for contradictions, and hiding the parity tokens again removes the direction. It reacts less to parity words outside a request.

![Both training runs zigzag fully for requests about a dog, the ocean, or a contradiction; the second run reacts less to "do not", to a question about an even number, and not at all to a fun fact about parity or to a list of even then odd numbers](figures/replica.png)

So the trained model moves an internal representation on request, layer by layer, without saying so in its reply. The prompts that ask for something else show what it responds to: the tokens "even" and "odd", their order, and the up and down verbs around them, with little regard for the concept named, for negation, or for whether any request was made. A test of whether a model controls its own activations needs prompts that separate the keywords from the meaning.

See all code, data and results here:
https://github.com/SauersML/zigzag
