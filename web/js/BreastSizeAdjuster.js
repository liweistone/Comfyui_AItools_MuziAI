app.registerExtension({
    name: "comfy.BreastSizeAdjuster",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.name === "BreastSizeAdjuster") {
            const orig_nodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                orig_nodeCreated?.apply(this, arguments);
                
                // 添加自定义UI元素
                const slider = this.widgets.find(w => w.name === "size_strength");
                if (slider) {
                    slider.inputEl.style.backgroundColor = "#ffccf5";
                }
            };
        }
    },
});