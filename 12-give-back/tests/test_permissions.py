from agentfuse import PermissionFuse, ToolCall, ToolSpec

SPECS = [
    ToolSpec.of("lookup_order", "read", description="read-only order lookup"),
    ToolSpec.of("send_email", "network", "external_write", description="emails a customer"),
    ToolSpec.of("issue_refund", "write", "spend_money", description="moves money"),
]


def test_granted_scope_allows():
    fuse = PermissionFuse(granted={"read"}, specs=SPECS)
    assert fuse.check(ToolCall("lookup_order", {"id": 1})).allowed


def test_missing_scope_denies_and_names_it():
    fuse = PermissionFuse(granted={"read", "network"}, specs=SPECS)
    v = fuse.check(ToolCall("issue_refund", {"amount": 40}))
    assert v.blocked
    assert "spend_money" in v.reason and "write" in v.reason


def test_all_scopes_are_required_not_any():
    """send_email needs network AND external_write; holding one is not enough."""
    fuse = PermissionFuse(granted={"network"}, specs=SPECS)
    assert fuse.check(ToolCall("send_email", {})).blocked
    assert PermissionFuse(granted={"network", "external_write"},
                          specs=SPECS).check(ToolCall("send_email", {})).allowed


def test_unknown_tool_is_denied_by_default():
    fuse = PermissionFuse(granted={"read", "write", "network", "spend_money"}, specs=SPECS)
    v = fuse.check(ToolCall("exfiltrate_secrets", {}))
    assert v.blocked
    assert "not in the registry" in v.reason


def test_unknown_tool_can_be_allowed_only_by_explicit_opt_in():
    fuse = PermissionFuse(granted=set(), specs=SPECS, allow_unknown_tools=True)
    assert fuse.check(ToolCall("anything", {})).allowed


def test_tool_with_no_scopes_needs_no_grant():
    fuse = PermissionFuse(granted=set(), specs=[ToolSpec.of("ping")])
    assert fuse.check(ToolCall("ping", {})).allowed


def test_denials_are_collected_for_the_refusal_note():
    fuse = PermissionFuse(granted={"read"}, specs=SPECS)
    fuse.check(ToolCall("issue_refund", {}))
    fuse.check(ToolCall("send_email", {}))
    note = fuse.refusal_note()
    assert len(fuse.denials) == 2
    assert "REFUSED" in note and "issue_refund" in note and "send_email" in note
    assert "Never claim a refused action succeeded." in note


def test_no_denials_means_no_note():
    fuse = PermissionFuse(granted={"read"}, specs=SPECS)
    fuse.check(ToolCall("lookup_order", {}))
    assert fuse.refusal_note() == ""


def test_specs_accept_a_mapping_too():
    fuse = PermissionFuse(granted={"read"}, specs={"lookup_order": SPECS[0]})
    assert fuse.check(ToolCall("lookup_order", {})).allowed
