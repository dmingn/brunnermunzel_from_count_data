# brunnermunzel_from_count_data

Compute the Brunner-Munzel test on two count datas (mappings from values to its occurence counts).

```
>>> from brunnermunzel_from_count_data import brunnermunzel_from_count_data
>>> x1 = {1: 11, 2: 2, 4: 1}
>>> x2 = {1: 3, 2: 1, 3: 4, 4: 2, 5: 1}
>>> w, p_value = brunnermunzel_from_count_data(x1, x2)
>>> w
3.1374674823029505
>>> p_value
0.005786208666151538
```

## Install

```
pip install git+https://github.com/dmingn/brunnermunzel_from_count_data.git
```
