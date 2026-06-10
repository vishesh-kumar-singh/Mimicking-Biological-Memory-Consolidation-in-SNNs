As per suggestions from reviewers at Neurocomputing journal, we need to cover the following:


1. **Test true synaptic freezing (Addressed via paper framing):** Reference--"The authors claim biological plausibility for mimicking synaptic consolidation, but implement a neuron-level P-factor where all incoming synapses of a neuron share the same value. This discards a core property of biological synapses: different synapses on the same neuron can be independently tagged and stabilized. The method actually selects and freezes important neurons, not synapses, which is closer to modular network engineering than to true synaptic consolidation. The claimed biological grounding is therefore overstated." 
   *Resolution:* We will update the paper's framing to clarify that the Neuron-Centric approximation (dendritic scaling) is a necessary mathematical abstraction for efficient SNNs ($O(N)$ instead of $O(N^2)$), and tone down the strict "synaptic" biological claims.


2. **Need to explain the random ablation results:** Reference--"The paper ignores the inherent sparse property of SNNs. With typical firing rates of 10-30%, about 70-80% of hidden neurons are largely silent. Randomly freezing 40% of neurons disrupts only about 40% of the truly active ones, leaving approximately 60% intact. This mathematically explains the observed 41% retention without invoking "effective parameter isolation." The authors fail to account for this necessity, leading to a biased interpretation of the random freezing baseline and an overestimation of the P-factor's contribution."
    *Resolution:* Updated in paper


3. **Theoretical Justification for reset:** Reference--"The "homeostatic reset" of novice neurons is analogized to biological neurogenesis, but large-scale random weight resetting does not occur in mature brains and is functionally closer to network damage. Moreover, neurons deemed "novice" were previously penalized by LTD as ineffective or noisy. The authors provide no mechanistic explanation for why randomly resetting such connections would benefit new learning. This operation lacks clear theoretical justification."


4. **Slight Wordings changes:** Reference--"The introduction states that "the important weights for previous tasks are altered" during new learning, implying an active mechanism targeting important weights. In fact, gradient descent updates weights indiscriminately based on the new task's loss, without considering their importance to previous tasks. This imprecise anthropomorphic wording misrepresents the mechanism of catastrophic forgetting."


5. **Need to improve Benchmarks and Continual Splits**: Reference--"The experimental results are mainly based on Split-MNIST, which is a relatively small and simple benchmark. Although Split-MNIST is useful as a preliminary proof of concept, it is not sufficient to demonstrate the generality and robustness of the proposed method for continual learning in SNNs. The authors should evaluate the method on more challenging datasets and task sequences, such as CIFAR-10/100, or other neuromorphic/event-based benchmarks. Testing on longer task sequences, rather than only two tasks, would also be important to show whether the proposed consolidation mechanism scales to realistic continual learning scenarios."


6. **Baselines Comparison**: Reference--"The current experiments compare the proposed method mainly with fine-tuning and internal ablation baselines such as random consolidation and fixed-index consolidation. These comparisons show that the P-Factor selection strategy is useful, but they do not establish the competitiveness of the method against existing SNN continual learning approaches. The authors should include quantitative comparisons with representative SNN continual learning methods, including regularization-based, replay-based, parameter-isolation, and biologically inspired SNN learning strategies. Without such comparisons, it is difficult to judge whether the proposed method provides a meaningful improvement over the state of the art."


7. **Better Numbers and Energy Analysis, longer horizon testing: Reference**--"The paper reports retention on Task A and accuracy on Task B, but a more complete continual learning evaluation should include standard metrics such as average accuracy over all tasks, backward transfer/forgetting, forward transfer, memory overhead, computational cost, and possibly energy-related metrics that are particularly relevant for SNNs. Since the method freezes part of the network and resets other neurons, it would also be helpful to analyze the remaining capacity after each task and how performance changes as the number of tasks increases."


8. **Hyper-parameter Sensitivity Analysis**: Reference--"The method relies on several important hyperparameters, including the consolidation threshold, potentiation/depression rates, and the P-Factor scaling term. The manuscript reports some threshold results, but the method appears sensitive to these choices and to random initialization. A more systematic sensitivity analysis is needed. The authors should clarify how these hyperparameters are selected, whether they are tuned on the test tasks, and whether the same settings work across different datasets and architectures."


9. **More Grounded Biological Analysis**: Refrence--"The current network is a relatively simple fully connected SNN. It is unclear whether the proposed P-Factor mechanism works similarly for convolutional SNNs, recurrent SNNs, deeper architectures, or event-based sensory streams. Since the paper claims relevance to neuromorphic computing, experiments on more realistic SNN architectures and an analysis of memory/computation overhead would strengthen the manuscript."


10. **Paper terms tone down and Proof Reading**: Reference--"The current network is a relatively simple fully connected SNN. It is unclear whether the proposed P-Factor mechanism works similarly for convolutional SNNs, recurrent SNNs, deeper architectures, or event-based sensory streams. Since the paper claims relevance to neuromorphic computing, experiments on more realistic SNN architectures and an analysis of memory/computation overhead would strengthen the manuscript."


11. **Need to mention we are using Class incremental Learning accuracy as evaluation metric.**