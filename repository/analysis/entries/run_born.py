from aiida import orm
from ..born_charges.born_analyzer import BornAnalyzer

def run(node_pk: int, **kwargs):
    """
    Standardized entry point for Born Charges and Elasticity analysis.
    """
    try:
        node = orm.load_node(node_pk)
    except Exception as e:
        return {"error": f"Failed to load node {node_pk}: {str(e)}"}
        
    if not isinstance(node, orm.WorkChainNode):
        return {"error": f"Node {node_pk} is not a WorkChainNode."}
        
    analyzer = BornAnalyzer(node)
    
    # Process options
    tolerance = kwargs.get('stability_tolerance', -5.0)
    
    # Run analysis
    results = analyzer.run_all()
    
    return results

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python run_born.py <node_pk>")
        sys.exit(1)
        
    pk = int(sys.argv[1])
    res = run(pk)
    print(json.dumps(res, indent=2))
