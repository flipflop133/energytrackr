"""JavaSetupStage: A specialized stage for setting up Java environment variables."""

import logging
import xml.etree.ElementTree as ET
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class JavaSetupStage(PipelineStage):
    """A specialized stage that sets JAVA_HOME or does other Java-specific tasks."""

    def run(self, _context: dict[str, Any]) -> None:
        """Sets up the Java environment variables for the given commit.

        Extracts the Java version from the project's pom.xml, maps it to the
        corresponding JAVA_HOME path, and sets the environment variables.

        Args:
            context: A dictionary containing the current execution context.
        """
        repo_path = Config.get_config().repo_path
        if not repo_path:
            logging.error("Repository path is not set in the configuration.")
            logging.error("Skipping Java setup stage. Defaulting to system Java.")
            return

        version = self.extract_java_version("pom.xml")
        if not version:
            logging.error("Valid Java version not found. Skipping Java setup stage.")
            return

        java_home = self.map_version_to_home(version)
        logging.info("Setting up Java environment with JAVA_HOME: %s", java_home)
        run_command(f"export JAVA_HOME={java_home}")
        run_command("export PATH=$JAVA_HOME/bin:$PATH")

    @staticmethod
    def map_version_to_home(version: str) -> str:
        """Maps a Java version string to the corresponding JAVA_HOME path.

        For example, '1.8' is mapped to '/usr/lib/jvm/java-8-openjdk'.
        If the version does not start with '1.', it is used directly.
        """
        # Convert version like "1.8" to "8"
        version_number = version.split(".")[1] if version.startswith("1.") and "." in version else version
        return f"/usr/lib/jvm/java-{version_number}-openjdk"

    @staticmethod
    def extract_java_version(pom_file: str) -> str | None:
        """Extracts the Java version from a Maven POM file.

        Args:
            pom_file (str): Path to the POM file.

        Returns:
            Optional[str]: The Java version specified in the POM file, or None if not found.
        """
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()

            # Handle XML namespaces (POM files typically include a namespace)
            ns = JavaSetupStage._get_xml_namespace(root)

            # Try extracting the Java version from <properties>
            java_version = JavaSetupStage._extract_from_properties(root, ns)
            if java_version:
                return java_version

            # Try extracting the Java version from maven-compiler-plugin
            java_version = JavaSetupStage._extract_from_compiler_plugin(root, ns)
            if java_version:
                return java_version

        except Exception:
            logging.exception("Error parsing pom.xml: %s", pom_file)
            return None
        else:
            logging.error("Java version not found in pom.xml.")
            return None

    @staticmethod
    def _get_xml_namespace(root: ET.Element) -> dict[str, str]:
        """Extracts the XML namespace from the root element."""
        if root.tag.startswith("{"):
            uri = root.tag.split("}")[0].strip("{")
            return {"ns": uri}
        return {}

    @staticmethod
    def _extract_from_properties(root: ET.Element, ns: dict[str, str]) -> str | None:
        """Extracts the Java version from the <properties> section."""
        properties = root.find("ns:properties", ns)
        if properties is not None:
            java_version = properties.find("ns:java.version", ns)
            if java_version is not None and java_version.text:
                return java_version.text.strip()
        return None

    @staticmethod
    def _extract_from_compiler_plugin(root: ET.Element, ns: dict[str, str]) -> str | None:
        """Extracts the Java version from the maven-compiler-plugin configuration."""
        plugins = root.findall(".//ns:plugin", ns)
        for plugin in plugins:
            if JavaSetupStage._is_maven_compiler_plugin(plugin, ns):
                return JavaSetupStage._get_java_version_from_plugin(plugin, ns)
        return None

    @staticmethod
    def _is_maven_compiler_plugin(plugin: ET.Element, ns: dict[str, str]) -> bool:
        """Checks if the given plugin is the maven-compiler-plugin."""
        artifact_id = plugin.find("ns:artifactId", ns)
        return artifact_id is not None and artifact_id.text is not None and artifact_id.text.strip() == "maven-compiler-plugin"

    @staticmethod
    def _get_java_version_from_plugin(plugin: ET.Element, ns: dict[str, str]) -> str | None:
        """Extracts the Java version from the maven-compiler-plugin's configuration."""
        configuration = plugin.find("ns:configuration", ns)
        if configuration is not None:
            source = configuration.find("ns:source", ns)
            target = configuration.find("ns:target", ns)
            # Return source version if available; otherwise, target version.
            if source is not None and source.text:
                return source.text.strip()
            if target is not None and target.text:
                return target.text.strip()
        return None
