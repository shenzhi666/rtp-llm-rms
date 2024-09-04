load("//:def.bzl", "copts", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
]

test_envs = {
    "TEST_USING_DEVICE": "ARM",
}

test_tags = [
    "arm",
]

cc_library(
    name = "arm_test_util",
    hdrs = [
        "ArmTestUtils.h",
    ],
    srcs = [],
    deps = [
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

test_deps = [
    "//src/fastertransformer/devices/arm_impl:arm_cpu_impl",
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/base_tests:base_tests",
    ":arm_test_util",
] + torch_deps()


cc_test(
    name = "arm_layernorm_op_test",
    srcs = [
        "ops/LayerNormOpTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    env = test_envs,
    tags = test_tags,
)


cc_test(
    name = "arm_embedlkp_op_test",
    srcs = [
        "ops/EmbeddingLookupTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    env = test_envs,
    tags = test_tags,
)
