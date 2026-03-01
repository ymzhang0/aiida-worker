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
