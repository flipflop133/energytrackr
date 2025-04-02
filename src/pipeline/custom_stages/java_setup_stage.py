"""JavaSetupStage: A specialized stage for setting up Java environment variables."""

import logging
import xml.etree.ElementTree as ET
from typing import Any

from config.config_store import Config
from pipeline.stage_interface import PipelineStage
from utils import run_command


class JavaSetupStage(PipelineStage):
    """A specialized stage that sets JAVA_HOME or does other Java-specific tasks."""

    def run(self, context: dict[str, Any]) -> None:
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

        version = self.extract_java_version(repo_path + "/pom.xml")
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
            ns = {}
            if root.tag.startswith("{"):
                uri = root.tag.split("}")[0].strip("{")
                ns = {"ns": uri}

            # First, try to find <java.version> in the <properties> section
            properties = root.find("ns:properties", ns)
            if properties is not None:
                java_version = properties.find("ns:java.version", ns)
                if java_version is not None and java_version.text:
                    return java_version.text.strip()

            # Alternatively, check for maven-compiler-plugin settings
            plugins = root.findall(".//ns:plugin", ns)
            for plugin in plugins:
                artifact_id = plugin.find("ns:artifactId", ns)
                if artifact_id is not None and artifact_id.text.strip() == "maven-compiler-plugin":
                    configuration = plugin.find("ns:configuration", ns)
                    if configuration is not None:
                        source = configuration.find("ns:source", ns)
                        target = configuration.find("ns:target", ns)
                        # Return source version if available; otherwise, target version.
                        if source is not None and source.text:
                            return source.text.strip()
                        elif target is not None and target.text:
                            return target.text.strip()

            logging.error("Java version not found in pom.xml.")

        except Exception:
            logging.exception("Error parsing pom.xml: %s")
            return None
        else:
            logging.error("Unexpected error while extracting Java version.")
            return None
