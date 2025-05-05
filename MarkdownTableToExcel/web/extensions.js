app.registerExtension({
    name: "comfy.markdown_table_to_excel",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.name === "MarkdownTableToExcel") {
            // 可以添加自定义前端逻辑
        }
    },
});