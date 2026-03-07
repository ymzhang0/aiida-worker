from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class WorkflowInputsRequest(BaseModel):
    entry_point: str = Field(..., description="AiiDA workflow entry point name")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Workflow input values")


class SpecResponse(BaseModel):
    entry_point: str
    inputs: dict[str, Any]


class ValidationResponse(BaseModel):
    success: bool
    message: str
    errors: list[dict[str, Any]] = Field(default_factory=list)
    entry_point: str | None = None
    missing_ports: list[str] = Field(default_factory=list)
    signature: list[dict[str, Any]] = Field(default_factory=list)
    builder_inputs: dict[str, Any] = Field(default_factory=dict)
    pseudo_expectations: list[dict[str, Any]] = Field(default_factory=list)
    recovery_plan: dict[str, Any] = Field(default_factory=dict)


class JobValidationRequest(BaseModel):
    entry_point: str = Field(..., description="AiiDA workflow entry point name")
    input_pks: dict[str, Any] = Field(default_factory=dict, description="Input node PK payload")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional non-node input parameters")


class JobValidationResponse(BaseModel):
    success: bool
    dry_run: bool = True
    entry_point: str
    summary: dict[str, Any]
    errors: list[dict[str, Any]] = Field(default_factory=list)


class SubmitResponse(BaseModel):
    pk: int
    uuid: str
    state: str


class SystemCountsResponse(BaseModel):
    computers: int
    codes: int
    workchains: int


class SystemInfoResponse(BaseModel):
    profile: str
    counts: SystemCountsResponse
    daemon_status: bool


class ComputerResource(BaseModel):
    label: str
    hostname: str
    description: str | None = None


class CodeResource(BaseModel):
    label: str
    default_plugin: str | None = None
    computer_label: str | None = None


class ResourcesResponse(BaseModel):
    computers: list[ComputerResource] = Field(default_factory=list)
    codes: list[CodeResource] = Field(default_factory=list)


class ProfileSwitchRequest(BaseModel):
    profile: str


class ArchiveLoadRequest(BaseModel):
    path: str


class ContextNodesRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)


class GroupCreateRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)


class GroupRenameRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)


class GroupAddNodesRequest(BaseModel):
    node_pks: list[int] = Field(default_factory=list)


class NodeSoftDeleteRequest(BaseModel):
    deleted: bool = True


class NodeScriptResponse(BaseModel):
    pk: int
    node_type: str
    language: str = "python"
    script: str


class PythonScriptRequest(BaseModel):
    script: str


class ScriptRegisterRequest(BaseModel):
    script_name: str | None = Field(default=None, description="Stable identifier of the script")
    skill_name: str | None = Field(default=None, description="Deprecated alias of script_name")
    script: str = Field(..., description="Python source code defining main(params)")
    description: str | None = Field(default=None, description="Optional short description")
    overwrite: bool = Field(default=True, description="Whether an existing script may be replaced")

    @model_validator(mode="after")
    def _normalize_name_fields(self) -> "ScriptRegisterRequest":
        if not self.script_name and not self.skill_name:
            raise ValueError("Either 'script_name' or 'skill_name' is required")
        if self.script_name and self.skill_name and self.script_name != self.skill_name:
            raise ValueError("'script_name' and 'skill_name' must match when both are provided")
        if not self.script_name:
            self.script_name = self.skill_name
        return self


class ScriptExecuteRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict, description="JSON params passed into main(params)")


class BuilderDraftRequest(BaseModel):
    workchain: str | None = Field(default=None, description="Deprecated alias of entry_point")
    entry_point: str | None = Field(default=None, description="Workflow entry point name")
    protocol: str | None = "moderate"
    intent_data: dict[str, Any] = Field(default_factory=dict, description="Intent payload mapped to protocol args")
    overrides: dict[str, Any] = Field(default_factory=dict)
    structure_pk: int | None = Field(default=None, description="Deprecated: moved into intent_data")
    code: str | None = Field(default=None, description="Deprecated: moved into intent_data")

    @model_validator(mode="after")
    def _normalize_builder_fields(self) -> "BuilderDraftRequest":
        if not self.entry_point and not self.workchain:
            raise ValueError("Either 'entry_point' or 'workchain' is required")
        if self.entry_point and self.workchain and self.entry_point != self.workchain:
            raise ValueError("'entry_point' and 'workchain' must match when both are provided")
        if not self.entry_point:
            self.entry_point = self.workchain
        if self.structure_pk is not None and "structure_pk" not in self.intent_data:
            self.intent_data["structure_pk"] = self.structure_pk
        if self.code is not None and "code" not in self.intent_data:
            self.intent_data["code"] = self.code
        return self


class BuilderSubmitRequest(BaseModel):
    draft: dict[str, Any]


class SubmissionScriptRequest(BaseModel):
    workchain: str | None = Field(default=None, description="Deprecated alias of entry_point")
    entry_point: str | None = Field(default=None, description="Workflow entry point name")
    protocol: str | None = "moderate"
    intent_data: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_script_fields(self) -> "SubmissionScriptRequest":
        if not self.entry_point and not self.workchain:
            raise ValueError("Either 'entry_point' or 'workchain' is required")
        if self.entry_point and self.workchain and self.entry_point != self.workchain:
            raise ValueError("'entry_point' and 'workchain' must match when both are provided")
        if not self.entry_point:
            self.entry_point = self.workchain
        return self


class BuilderDraftResponse(BaseModel):
    success: bool
    status: str
    entry_point: str
    protocol: str | None = None
    intent_data: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    signature: list[dict[str, Any]] = Field(default_factory=list)
    pseudo_expectations: list[dict[str, Any]] = Field(default_factory=list)
    builder_inputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    missing_ports: list[str] = Field(default_factory=list)
    recovery_plan: dict[str, Any] = Field(default_factory=dict)
    preview: str | None = None


class BuilderScriptResponse(BaseModel):
    entry_point: str
    protocol: str | None = None
    signature: list[dict[str, Any]] = Field(default_factory=list)
    script: str


class ProcessCloneDraftResponse(BaseModel):
    process_label: str
    entry_point: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    recommended_inputs: dict[str, Any] = Field(default_factory=dict)
    advanced_settings: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

class InfrastructureSetupRequest(BaseModel):
    # Computer fields
    computer_label: str = Field(..., min_length=1, max_length=255)
    hostname: str = Field(..., min_length=1)
    computer_description: str | None = None
    transport_type: str = "core.ssh"
    scheduler_type: str = "core.direct"
    work_dir: str = "/tmp/aiida"
    mpiprocs_per_machine: int = 1
    mpirun_command: str = "mpirun -np {tot_num_mpiprocs}"
    prepend_text: str | None = None
    append_text: str | None = None
    
    # Auth fields
    username: str | None = None
    key_filename: str | None = None
    proxy_command: str | None = None
    proxy_jump: str | None = None
    safe_interval: float | None = None
    use_login_shell: bool = True
    connection_timeout: int | None = None
    
    # Code fields
    code_label: str | None = None
    code_description: str | None = None
    default_calc_job_plugin: str | None = None
    remote_abspath: str | None = None
    code_prepend_text: str | None = None
    code_append_text: str | None = None

class SSHHostDetails(BaseModel):
    alias: str
    hostname: str | None = None
    username: str | None = None
    port: int | None = None
    proxy_jump: str | None = None
    proxy_command: str | None = None
    identity_file: str | None = None


class UserInfoResponse(BaseModel):
    first_name: str
    last_name: str
    email: str
    institution: str


class ProfileSetupRequest(BaseModel):
    profile_name: str
    first_name: str
    last_name: str
    email: str
    institution: str
    filepath: str
    backend: str = "core.sqlite_dos"
    set_as_default: bool = True

class CodeSetupRequest(BaseModel):
    computer_label: str
    label: str
    description: str | None = None
    default_calc_job_plugin: str
    remote_abspath: str
    prepend_text: str | None = None
    append_text: str | None = None
    with_mpi: bool = True
    use_double_quotes: bool = False

class CodeDetailedResponse(BaseModel):
    pk: int
    label: str
    description: str | None = None
    default_calc_job_plugin: str
    remote_abspath: str
    prepend_text: str | None = None
    append_text: str | None = None
    with_mpi: bool
    use_double_quotes: bool
