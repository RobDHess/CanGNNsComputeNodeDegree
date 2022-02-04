# CanGNNsComputeNodeDegree?

While I know that simple MPNNs are able to compute the node degree with sum aggregation and a single layer, all I had ever seen were theoretical proofs. In my experience, what neural networks are able to learn and what they actually learn can be two different things. So I created this repository to check and see for myself. The answer turns out to be: yes, they can!

To run this, select one of the datasets—or add your own!—then add a model—or add your own!—and run the following command:

```
python3 main.py --dataset=PROTEINS --model=MPNN --aggr=add --num_workers=0
```

or, if you like to see the model fail:

```
python3 main.py --dataset=PROTEINS --model=MPNN --aggr=mean --num_workers=0
```



### Dependencies
This repository uses 
* PyTorch
* PyTorch Geometric
* PyTorch Lightning
