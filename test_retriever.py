# test_retriever.py
from retriever.retriever import Retriever

def test_retriever():
    # Initialize
    retriever = Retriever()
    
    # Add documents
    retriever.add_documents(["doc1.txt", "doc2.md"])
    
    # Test queries
    queries = [
        "What is machine learning?",
        "Python uses in data science",
        "Deep learning frameworks"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.query(query)
        for i, res in enumerate(results):
            print(f"{i+1}. [Score: {res['score']:.3f}] {res['text'][:100]}...")
    
    # Save and load test
    retriever.save("test_index")
    loaded_retriever = Retriever.load("test_index")
    assert len(loaded_retriever.documents) == len(retriever.documents)
    print("\nSave/load test passed!")

if __name__ == "__main__":
    test_retriever()