# Vscode setting
## press [Ctrl + ,], opening settings(json)
### replace everything with:

{
    // 1. 控制快速建议：仅在非注释/字符串中开启，但延迟调高避免误触发
    "editor.quickSuggestions": {
        "other": true,
        "comments": false,
        "strings": false
    },
    "editor.quickSuggestionsDelay": 500,

    // 2. 关键：保留触发字符（如(、.）的提示，这样函数参数会正常弹出
    "editor.suggestOnTriggerCharacters": true,

    // 3. 保留当前文件内的单词联想
    "editor.wordBasedSuggestions": "matchingDocuments",

    // 4. 关闭AI内联建议（Copilot等）
    "editor.inlineSuggest.enabled": false,
    "github.copilot.enable": {
        "*": false
    },

    // 5. 关闭自动接受补全，避免误提交
    "editor.acceptSuggestionOnCommitCharacter": false,
    "editor.acceptSuggestionOnEnter": "off",

    // 6. 关闭代码片段、自动导入等干扰项
    "editor.snippetSuggestions": "none",
    "javascript.suggest.autoImports": false,
    "typescript.suggest.autoImports": false,
    "workbench.colorTheme": "Dark Modern"
}
