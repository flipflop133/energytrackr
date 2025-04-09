"""JavaSetupStage: A specialized stage for setting up Java environment variables."""

import xml.etree.ElementTree as ET
from typing import Any

from pipeline.stage_interface import PipelineStage
from utils.logger import logger
from utils.utils import run_command


class JavaSetupStage(PipelineStage):
    """A specialized stage that sets JAVA_HOME or does other Java-specific tasks."""

    def run(self, context: dict[str, Any]) -> None:
        """Sets up the Java environment variables for the given commit.

        Extracts the Java version from the project's pom.xml, maps it to the
        corresponding JAVA_HOME path, and sets the environment variables.

        Args:
            context: A dictionary containing the current execution context.
        """
        repo_path = context.get("repo_path")
        if not repo_path:
            logger.error("Repository path is not set in the configuration.", context=context)
            logger.error("Skipping Java setup stage. Defaulting to system Java.", context=context)
            return

        version = self.extract_java_version("pom.xml", context)
        if not version:
            logger.error("Valid Java version not found. Skipping Java setup stage.", context=context)
            return

        java_home = self.map_version_to_home(version)
        logger.info(f"Setting up Java environment with JAVA_HOME: {java_home}", context=context)
        run_command(f"export JAVA_HOME={java_home}", context=context)
        run_command("export PATH=$JAVA_HOME/bin:$PATH", context=context)

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
    def extract_java_version(pom_file: str, context: dict[str, Any]) -> str | None:
        """Extracts the Java version from a Maven POM file.

        Attempts the following methods:
          1. Look for <java.version> in the <properties> section (backward compatibility).
          2. Extract a properties map and then find the Java version from the maven-compiler-plugin
             configuration (supporting <release>, <source>, and <target> tags with property resolution).
          3. Also check profiles for maven-compiler-plugin configuration.

        Args:
            pom_file (str): Path to the POM file.
            context (dict[str, Any]): The context dictionary for logging.

        Returns:
            Optional[str]: The Java version specified in the POM file, or None if not found.
        """
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()

            # Handle XML namespaces (POM files typically include a namespace)
            ns = JavaSetupStage._get_xml_namespace(root)

            # 1. Try extracting directly from <properties> with <java.version>
            java_version = JavaSetupStage._extract_from_properties(root, ns)
            if java_version:
                return java_version

            # 2. Extract all properties into a dictionary for property substitution
            properties_map = JavaSetupStage._extract_properties_map(root, ns)

            # 3. Try extracting the Java version from maven-compiler-plugin in the main build
            java_version = JavaSetupStage._extract_from_compiler_plugin(root, ns, properties_map)
            if java_version:
                return java_version

            # 4. Also check profiles for maven-compiler-plugin configuration
            java_version = JavaSetupStage._extract_from_profiles(root, ns, properties_map)
            if java_version:
                return java_version

        except Exception:
            logger.exception("Error parsing pom.xml: %s", pom_file, context=context)
            return None

        logger.error("Java version not found in pom.xml.", context=context)
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
        """Extracts the Java version from the <properties> section using the tag <java.version>."""
        properties = root.find("ns:properties", ns)
        if properties is not None:
            java_version = properties.find("ns:java.version", ns)
            if java_version is not None and java_version.text:
                return java_version.text.strip()
        return None

    @staticmethod
    def _extract_properties_map(root: ET.Element, ns: dict[str, str]) -> dict[str, str]:
        """Extracts all properties from the <properties> section into a dictionary."""
        properties = root.find("ns:properties", ns)
        result = {}
        if properties is not None:
            for child in properties:
                if child.text:
                    # Remove namespace if exists
                    tag = child.tag.split("}")[-1]
                    result[tag] = child.text.strip()
        return result

    @staticmethod
    def _extract_from_compiler_plugin(root: ET.Element, ns: dict[str, str], properties_map: dict[str, str]) -> str | None:
        """Extracts the Java version from the maven-compiler-plugin configuration in the main build.

        This method supports the <release> tag as well as <source> and <target> tags,
        with resolution of property placeholders.
        """
        plugins = root.findall(".//ns:plugin", ns)
        for plugin in plugins:
            if JavaSetupStage._is_maven_compiler_plugin(plugin, ns):
                version = JavaSetupStage._get_java_version_from_plugin(plugin, ns, properties_map)
                if version:
                    return version
        return None

    @staticmethod
    def _extract_from_profiles(root: ET.Element, ns: dict[str, str], properties_map: dict[str, str]) -> str | None:
        """Extracts the Java version from the maven-compiler-plugin configuration within profiles."""
        profiles = root.findall("ns:profiles/ns:profile", ns)
        for profile in profiles:
            build = profile.find("ns:build", ns)
            if build is None:
                continue
            plugins = build.find("ns:plugins", ns)
            if plugins is None:
                continue
            for plugin in plugins.findall("ns:plugin", ns):
                if JavaSetupStage._is_maven_compiler_plugin(plugin, ns):
                    version = JavaSetupStage._get_java_version_from_plugin(plugin, ns, properties_map)
                    if version:
                        return version
        return None

    @staticmethod
    def _is_maven_compiler_plugin(plugin: ET.Element, ns: dict[str, str]) -> bool:
        """Checks if the given plugin is the maven-compiler-plugin."""
        artifact_id = plugin.find("ns:artifactId", ns)
        return artifact_id is not None and artifact_id.text is not None and artifact_id.text.strip() == "maven-compiler-plugin"

    @staticmethod
    def _get_java_version_from_plugin(plugin: ET.Element, ns: dict[str, str], properties: dict[str, str]) -> str | None:
        """Extracts the Java version from the maven-compiler-plugin's configuration.

        Supports tags <release>, <source>, and <target>. Resolves property placeholders
        such as ${source.version} using the provided properties dictionary. It checks both
        direct configuration and configuration within executions.
        """
        # Check direct configuration element
        configuration = plugin.find("ns:configuration", ns)
        if configuration is not None:
            version = JavaSetupStage._get_version_from_configuration(configuration, ns, properties)
            if version:
                return version

        # Check executions if direct configuration is not found
        executions = plugin.find("ns:executions", ns)
        if executions is not None:
            for execution in executions.findall("ns:execution", ns):
                configuration = execution.find("ns:configuration", ns)
                if configuration is not None:
                    version = JavaSetupStage._get_version_from_configuration(configuration, ns, properties)
                    if version:
                        return version

        return None

    @staticmethod
    def _get_version_from_configuration(
        configuration: ET.Element,
        ns: dict[str, str],
        properties: dict[str, str],
    ) -> str | None:
        """Helper method to extract Java version from a configuration element."""
        for tag in ["release", "source", "target"]:
            version_el = configuration.find(f"ns:{tag}", ns)
            if version_el is not None and version_el.text:
                version = version_el.text.strip()
                # Resolve property if the version is in the form ${...}
                if version.startswith("${") and version.endswith("}"):
                    key = version[2:-1]
                    resolved = properties.get(key)
                    if resolved:
                        return resolved
                else:
                    return version
        return None
