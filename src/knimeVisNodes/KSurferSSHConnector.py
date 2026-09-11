import sys
import os
import uuid
import logging

import knime.extension as knext
import torch

torch.set_num_threads(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)
knimeVis_category = kutil.get_knimeVis_category()

_node_dir = os.path.dirname(os.path.realpath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

from ksurfer_core.connection_port import (  # noqa: E402
    KSurferConnectionSpec,
    KSurferConnectionPortObject,
    ksurfer_connection_port_type,
    store_session,
)


@knext.parameter_group(label="SSH Credentials")
class SshCredentials:
    host = knext.StringParameter(
        label="Host",
        description="Hostname or IP address of the HPC login node.",
        default_value="",
    )
    port = knext.IntParameter(
        label="Port", description="SSH port. Usually 22.",
        default_value=22, min_value=1, max_value=65535,
    )
    username = knext.StringParameter(
        label="Username", description="Your HPC account username.",
        default_value="",
    )
    auth_method = knext.StringParameter(
        label="Authentication method",
        description="`password` or `key`.",
        default_value="password", enum=["password", "key"],
    )
    key_path = knext.StringParameter(
        label="Private key path",
        description="Path to your SSH private key. Only used when auth method is `key`.",
        default_value="",
    ).rule(knext.OneOf(auth_method, ["password"]), knext.Effect.HIDE)
    password = knext.StringParameter(
        label="Password / key passphrase",
        description=(
            "Your HPC password or key passphrase. Used only to establish "
            "the SSH session. Never written to the workflow file."
        ),
        default_value="",
    )
    accept_new_host_key = knext.BoolParameter(
        label="Auto-accept unknown host keys",
        description="Enable only for initial setup. Use known_hosts in production.",
        default_value=False, is_advanced=True,
    )


@knext.node(
    name="SSH Connection (python)",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/remote-access.png",
    category=knimeVis_category,
    id="ksurfer-ssh-connector",
)
@knext.output_port(
    name="KSurfer SSH Connection",
    description="SSH session passed to downstream KSurfer nodes without re-authentication.",
    port_type=ksurfer_connection_port_type,
)
class KSurferSSHConnectorNode:
    """Establish the SSH connection to the HPC cluster.

    Authenticates once and stores credentials in a secure temp file so
    downstream nodes (Subject Validator, Feature Extractor, Hippo/Amygdala
    Reader) can reconnect without asking for the password again.
    """

    credentials = SshCredentials()

    def configure(self, configure_context):
        if not self.credentials.host:
            raise knext.InvalidParametersError("Host must not be empty.")
        if not self.credentials.username:
            raise knext.InvalidParametersError("Username must not be empty.")
        if self.credentials.auth_method == "key" and not self.credentials.key_path:
            raise knext.InvalidParametersError(
                "Private key path must not be empty when using key auth."
            )
        return KSurferConnectionSpec(
            session_id="not-yet-connected",
            host=self.credentials.host,
            port=self.credentials.port,
            username=self.credentials.username,
            auth_method=self.credentials.auth_method,
            key_path=self.credentials.key_path,
        )

    def execute(self, exec_context):
        _node_dir = os.path.dirname(os.path.realpath(__file__))
        if _node_dir not in sys.path:
            sys.path.insert(0, _node_dir)

        from ksurfer_core.ssh import open_client

        exec_context.set_progress(0.1, "Establishing SSH connection...")

        client = open_client(
            host=self.credentials.host,
            port=self.credentials.port,
            username=self.credentials.username,
            auth_method=self.credentials.auth_method,
            key_path=self.credentials.key_path,
            password=self.credentials.password or None,
            accept_new_host_key=self.credentials.accept_new_host_key,
        )

        # Verify exec channel works
        _, stdout, _ = client.exec_command("echo ok", timeout=10)
        stdout.channel.recv_exit_status()
        client.close()

        # Store credentials in temp file — downstream nodes read this file
        # to reconnect without needing a password field in their dialogs
        session_id = str(uuid.uuid4())
        store_session(
            session_id=session_id,
            host=self.credentials.host,
            port=self.credentials.port,
            username=self.credentials.username,
            auth_method=self.credentials.auth_method,
            key_path=self.credentials.key_path,
            password=self.credentials.password or "",
        )

        exec_context.set_progress(
            1.0,
            f"Connected to {self.credentials.username}@"
            f"{self.credentials.host}:{self.credentials.port}"
        )

        spec = KSurferConnectionSpec(
            session_id=session_id,
            host=self.credentials.host,
            port=self.credentials.port,
            username=self.credentials.username,
            auth_method=self.credentials.auth_method,
            key_path=self.credentials.key_path,
        )
        return KSurferConnectionPortObject(spec)
