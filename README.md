# t5 Zero-shot on Winogrande

Evaluate t5's pre-training objective on Winogrande.

Input:
```
The trophy did not fit into the suitcase because the <extra_id_0> was too large. </s>
```

Opt1:
```
<extra_id_0> trophy </s>
```

Opt2:
```
<extra_id_0> suitcase </s>
```

Take the opt with the lower perplexity to be the correct answer.

t5-11b dev results:
```
{"acc-xs": 0.6250986582478295, "acc-s": 0.6250986582478295, "acc-m": 0.6250986582478295, "acc-l": 0.6250986582478295, "acc-xl": 0.6250986582478295, "auc": 0.6250986582478295}
```
