1. 安装环境
```bash
conda create --name news python=3.10
conda activate news
pip install torch
pip install transformers==4.37.1
pip install accelerate==0.21.0
pip install evaluate
pip install rouge_score
pip install tqdm
pip install sentencepiece
pip install protobuf

```

2. 运行代码
```bash
python main.py
```