import asyncio
import plumpy.events
plumpy.events.set_event_loop_policy()
asyncio.set_event_loop(asyncio.new_event_loop())

from aiida import load_profile
load_profile()

from aiida.orm import load_node, load_code
from aiida.plugins import WorkflowFactory
from aiida.engine import run_get_node

PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
code = load_code('qe-750-pw@lucia')
structure = load_node(156)

overrides = {
    'base': {'pseudo_family': 'PseudoDojo/0.5/PBE/SR/standard/upf'},
    'base_final_scf': {'pseudo_family': 'PseudoDojo/0.5/PBE/SR/standard/upf'}
}

builder = PwRelaxWorkChain.get_builder_from_protocol(
    code=code,
    structure=structure,
    protocol='moderate',
    overrides=overrides
)

print("Starting run_get_node...")
result, node = run_get_node(builder)
print(f"Finished Workchain PK: {node.pk}")
