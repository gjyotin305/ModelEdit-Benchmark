## Instructions

### Instructions To Run:

Make folders to store the merged weight and result
```bash
mkdir nerual_merge
mkdir res
```

Train the experts
```bash
python main_experts.py
```

Train the Indexer
```bash
python indexer.py
```

Infer for Evaluation
```bash
python infer.py
```

