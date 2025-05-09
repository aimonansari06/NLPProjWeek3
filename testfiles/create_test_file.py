# create_test_files.py
with open("doc1.txt", "w") as f:
    f.write("Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.\nDeep learning is a specialized form of machine learning that uses neural networks with multiple layers.")

with open("doc2.md", "w") as f:
    f.write("""
# Python Programming
Python is an interpreted, high-level programming language known for its readability.
- Used in web development (Django, Flask)
- Popular for data science (Pandas, NumPy)
- Strong machine learning ecosystem (TensorFlow, PyTorch)
""")

# For PDF testing, you would need an actual PDF file