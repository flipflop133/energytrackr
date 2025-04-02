package com.example;

import org.junit.platform.engine.DiscoverySelector;
import org.junit.platform.engine.discovery.DiscoverySelectors;
import org.junit.platform.engine.support.descriptor.ClassSource;
import org.junit.platform.launcher.*;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;

import java.io.File;
import java.io.PrintWriter;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.*;

public class TestDiscovery {

    public static void main(String[] args) throws Exception {
        // Paths to the compiled test and main classes of the external project
        String testClassesPath = "projects/jsoup/.cachejsoup/target/test-classes";
        String mainClassesPath = "projects/jsoup/.cachejsoup/target/classes";

        File testClassesDir = new File(testClassesPath);
        File mainClassesDir = new File(mainClassesPath);
        if (!testClassesDir.exists() || !mainClassesDir.exists()) {
            System.err.println("❌ One of the required directories does not exist: " 
                + testClassesPath + " or " + mainClassesPath);
            return;
        }

        // Create a URLClassLoader that includes both test and main classes directories
        URLClassLoader externalClassLoader = new URLClassLoader(
            new URL[] {
                testClassesDir.toURI().toURL(),
                mainClassesDir.toURI().toURL()
            },
            Thread.currentThread().getContextClassLoader()
        );
        // Set the thread context class loader to our external class loader
        Thread.currentThread().setContextClassLoader(externalClassLoader);

        // Discover all test classes in the test classes directory
        List<Class<?>> candidateTestClasses = findAllClasses(testClassesDir, "");

        // Build selectors for each candidate test class
        List<DiscoverySelector> selectors = new ArrayList<>();
        for (Class<?> cls : candidateTestClasses) {
            selectors.add(DiscoverySelectors.selectClass(cls));
        }

        // Build the launcher discovery request using the selectors
        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(selectors)
                .build();

        Launcher launcher = LauncherFactory.create();
        TestPlan testPlan = launcher.discover(request);

        // Open a file to write the discovered tests
        try (PrintWriter writer = new PrintWriter(new File("discovered_tests.txt"))) {
            // Recursively write the discovered tests to the file
            for (TestIdentifier test : testPlan.getRoots()) {
                writeTestsRecursive(testPlan, test, writer);
            }
        }
    }

    // Recursive method to write tests in the TestPlan to a file
    private static void writeTestsRecursive(TestPlan testPlan, TestIdentifier identifier, PrintWriter writer) {
        for (TestIdentifier child : testPlan.getChildren(identifier)) {
            if (child.isTest()) {
                Optional<TestIdentifier> parent = testPlan.getParent(child);
                String className = parent
                    .flatMap(p -> p.getSource())
                    .filter(ClassSource.class::isInstance)
                    .map(s -> ((ClassSource) s).getClassName())
                    .orElse("UnknownClass");

                writer.println(className + "#" + child.getDisplayName().replace("()", ""));
            } else {
                writeTestsRecursive(testPlan, child, writer);
            }
        }
    }

    // Recursive method to find all .class files and convert them into Java classes
    private static List<Class<?>> findAllClasses(File baseDir, String packagePrefix) {
        List<Class<?>> result = new ArrayList<>();
        File[] files = baseDir.listFiles();
        if (files == null) return result;

        for (File file : files) {
            if (file.isDirectory()) {
                result.addAll(findAllClasses(file, packagePrefix + file.getName() + "."));
            } else if (file.getName().endsWith(".class") && !file.getName().contains("$")) {
                String className = packagePrefix + file.getName().replace(".class", "");
                try {
                    // Load using the thread context class loader
                    result.add(Class.forName(className, true, Thread.currentThread().getContextClassLoader()));
                } catch (ClassNotFoundException | NoClassDefFoundError e) {
                    System.err.println("Failed to load class: " + className + " (" + e.getMessage() + ")");
                }
            }
        }
        return result;
    }
}
