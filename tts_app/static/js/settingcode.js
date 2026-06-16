var configData = {};
var languageMappingData = {};
var homophoneMappingData = {};
var polyphonicMappingData = {};

$(document).ready(function () {
    $.ajax({
        url: '/admin/get_config',
        type: 'GET',
        dataType: 'json',
        contentType: 'application/json',
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function (response) {
            configData = response;
            languageMappingData = response.language_mapping || {};
            homophoneMappingData = response.homophone_mapping || {};
            polyphonicMappingData = response.polyphonic_mapping || {};
            show_config();
            render_language_mapping_list(languageMappingData);
            render_homophone_mapping_list(homophoneMappingData);
            render_polyphonic_mapping_list(polyphonicMappingData);
        },
        error: function () {
            alert("加载配置失败！");
        }
    });

    $(".config-save").click(function () {
        set_config();
    });

    $("#add-language-mapping").click(function () {
        add_language_mapping_from_inputs();
    });

    $("#save-language-mapping").click(function () {
        save_language_mapping();
    });

    $("#add-homophone-mapping").click(function () {
        add_homophone_mapping_from_inputs();
    });

    $("#save-homophone-mapping").click(function () {
        save_homophone_mapping();
    });

    $("#add-polyphonic-mapping").click(function () {
        add_polyphonic_mapping_from_inputs();
    });

    $("#save-polyphonic-mapping").click(function () {
        save_polyphonic_mapping();
    });

    $("#language-mapping-list").on("click", ".language-remove", function () {
        $(this).closest(".language-mapping-row").remove();
    });

    $("#homophone-mapping-list").on("click", ".homophone-remove", function () {
        $(this).closest(".homophone-mapping-row").remove();
    });

    $("#polyphonic-mapping-list").on("click", ".polyphonic-remove", function () {
        $(this).closest(".polyphonic-mapping-row").remove();
    });

    // Add new API key
    $('#add-api-key').click(function () {
        var newKey = {
            key: generateRandomKey(),
            enabled: true
        };
        configData.system.api_keys.push(newKey);
        addApiKeyHtml(newKey, configData.system.api_keys.length - 1);
    });

    // Remove API key
    $('#api-keys').on('click', '.btn-remove-key', function () {
        var index = $(this).closest('.input-group').data('index');
        configData.system.api_keys.splice(index, 1);

        $(this).closest('.input-group').remove();

        $('#api-keys .input-group').each(function (i) {
            $(this).attr('data-index', i);
            $(this).find('.input-group-text').text(`API Key ${i + 1}`);
        });
    });

    // Toggle API Key Enabled status
    $('#api-key-enabled').change(function () {
        configData.system.api_key_enabled = $(this).prop('checked');
    });
});

function renderApiKeys() {
    var container = $('#api-keys');
    container.empty();

    configData.system.api_keys.forEach(function (apiKey, index) {
        addApiKeyHtml(apiKey, index);
    });
}

function addApiKeyHtml(apiKey, index) {
    var container = $('#api-keys');

    var apiKeyHtml = `
        <div class="input-group mb-3" data-index="${index}">
            <span class="input-group-text">API Key ${index + 1}</span>
            <input type="text" class="form-control" value="${apiKey.key}" readonly>
            <button class="btn btn-danger btn-remove-key">Remove</button>
            <div class="form-check form-switch ms-2">
                <input class="form-check-input" type="checkbox" ${apiKey.enabled ? 'checked' : ''}>
                <label class="form-check-label">Enabled</label>
            </div>
        </div>
    `;
    container.append(apiKeyHtml);
}

function generateRandomKey() {
    var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var keyLength = 24;
    var result = '';
    for (var i = 0; i < keyLength; i++) {
        var randomIndex = Math.floor(Math.random() * characters.length);
        result += characters[randomIndex];
    }
    return result;
}

function show_config() {
    $('#vits-config').empty();
    $('#w2v2-vits-config').empty();
    $('#hubert-vits-config').empty();
    $('#bert-vits2-config').empty();
    $('#log-config').find('.item:not(:first)').remove();
    $('#tts-model-config').empty();

    $.each(configData.vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'vits-config-' + formattedKey;
        $('#vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    });

    $.each(configData.w2v2_vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'w2v2-vits-config-' + formattedKey;
        $('#w2v2-vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    });

    $.each(configData.hubert_vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'hubert-vits-config-' + formattedKey;
        $('#hubert-vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    });

    $.each(configData.bert_vits2_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'bert-vits2-config-' + formattedKey;
        var inputValue = (value === null) ? '' : value;
        $('#bert-vits2-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${inputValue}">
        </div>
        `);
    });

    $.each(configData.log_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (key != 'logging_level') {
            $('#log-config').append(`
            <div class="input-group mb-3 item">
                <span class="input-group-text">${key}</span>
                <input type="text" class="form-control" id="${formattedKey}" value="${value}">
            </div>
            `);
        }
    });

    $.each(configData.tts_model_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey !== "tts-models") {
            $('#tts-model-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${formattedKey}" value="${value}">
        </div>
        `);
        }
    });

    $.each(configData.language_identification, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        $('#language-identification ' + '#' + formattedKey).val(value);
    });

    $.each(configData.http_service, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey == 'api-key-enable' || formattedKey == 'debug') {
            $('#' + formattedKey).prop('checked', value);
        } else {
            $('#' + formattedKey).val(value);
        }
    });

    renderApiKeys();

    $.each(configData.system, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey == 'api-key-enabled' || formattedKey == 'cache-audio') {
            $('#' + formattedKey).prop('checked', value);
        } else {
            $('#' + formattedKey).val(value);
        }
    });

    $.each(configData.admin, function (key, value) {
        $('#' + key).val(value);
    });
}

function set_config() {
    configData = {}

    $('.configuration .form-label').each(function () {
        var labelId = $(this).next().attr('id');
        if (!labelId) {
            return;
        }

        var nestedDict = {};

        $('#' + labelId).find('.item').each(function () {
            var itemId = $(this).find('input, select').attr('id').replace(/-/g, '_');
            itemId = itemId.replace(labelId.replace(/-/g, '_') + "_", "");

            var itemValue;
            if ($(this).find('input').is(':checkbox')) {
                itemValue = $(this).find('input').prop('checked');
            } else {
                itemValue = $(this).find('input, select').val();

                if (itemValue === "") {
                    itemValue = null;
                }

                if (itemId === "language_automatic_detect") {
                    itemValue = itemValue ? itemValue.split(" ") : [];
                }
            }
            nestedDict[itemId] = itemValue;
        });

        if (labelId === "system") {
            nestedDict['api_keys'] = [];
            $('#api-keys .input-group').each(function () {
                var apiKey = $(this).find('input[type="text"]').val();
                var apiKeyEnabled = $(this).find('.form-check-input').is(':checked');
                nestedDict['api_keys'].push({
                    key: apiKey,
                    enabled: apiKeyEnabled
                });
            });
        }

        configData[labelId.replace(/-/g, '_')] = nestedDict;
    });

    $.ajax({
        type: "POST",
        url: "/admin/set_config",
        data: JSON.stringify(configData),
        contentType: "application/json",
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function () {
            alert("主配置已保存");
        },
        error: function () {
            alert("保存主配置时出错，请查看日志！");
        }
    });

    return configData;
}

function render_language_mapping_list(mapping) {
    var container = $("#language-mapping-list");
    container.empty();

    var entries = Object.entries(mapping || {});
    if (entries.length === 0) {
        container.append('<div class="text-muted">暂无自定义词条</div>');
        return;
    }

    entries.forEach(function (entry) {
        container.append(create_language_mapping_row(entry[0], entry[1]));
    });
}

function create_language_mapping_row(term, lang) {
    var row = $(`
        <div class="language-mapping-row">
            <input type="text" class="form-control language-term" placeholder="自定义词条">
            <input type="text" class="form-control language-lang" placeholder="语言标签">
            <button type="button" class="btn btn-outline-danger language-remove">删除</button>
        </div>
    `);

    row.find(".language-term").val(term || "");
    row.find(".language-lang").val(lang || "");
    return row;
}

function add_language_mapping_from_inputs() {
    var term = $("#language-mapping-term").val().trim();
    var lang = $("#language-mapping-lang").val().trim().toLowerCase();

    if (!term || !lang) {
        alert("请先填写词条和语言标签。");
        return;
    }

    $("#language-mapping-list .text-muted").remove();
    $("#language-mapping-list").append(create_language_mapping_row(term, lang));
    $("#language-mapping-term").val("");
    $("#language-mapping-lang").val("");
}

function collect_language_mapping() {
    var mapping = {};

    $("#language-mapping-list .language-mapping-row").each(function () {
        var term = $(this).find(".language-term").val().trim();
        var lang = $(this).find(".language-lang").val().trim().toLowerCase();

        if (term && lang) {
            mapping[term] = lang;
        }
    });

    return mapping;
}

function save_language_mapping() {
    var mapping = collect_language_mapping();

    $.ajax({
        type: "POST",
        url: "/admin/set_language_mapping",
        data: JSON.stringify({ language_mapping: mapping }),
        contentType: "application/json",
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function () {
            languageMappingData = mapping;
            alert("自定义词条已保存");
        },
        error: function () {
            alert("保存自定义词条失败，请查看日志！");
        }
    });
}

function render_homophone_mapping_list(mapping) {
    var container = $("#homophone-mapping-list");
    container.empty();

    var entries = Object.entries(mapping || {});
    if (entries.length === 0) {
        container.append('<div class="text-muted">暂无谐音映射</div>');
        return;
    }

    entries.forEach(function (entry) {
        container.append(create_homophone_mapping_row(entry[0], entry[1]));
    });
}

function create_homophone_mapping_row(term, value) {
    var row = $(`
        <div class="homophone-mapping-row language-mapping-row">
            <input type="text" class="form-control homophone-term" placeholder="原词或符号">
            <input type="text" class="form-control homophone-value" placeholder="替换后的谐音词">
            <button type="button" class="btn btn-outline-danger homophone-remove">删除</button>
        </div>
    `);

    row.find(".homophone-term").val(term || "");
    row.find(".homophone-value").val(value || "");
    return row;
}

function add_homophone_mapping_from_inputs() {
    var term = $("#homophone-mapping-term").val().trim();
    var value = $("#homophone-mapping-value").val().trim();

    if (!term || !value) {
        alert("请先填写原词和谐音词。");
        return;
    }

    $("#homophone-mapping-list .text-muted").remove();
    $("#homophone-mapping-list").append(create_homophone_mapping_row(term, value));
    $("#homophone-mapping-term").val("");
    $("#homophone-mapping-value").val("");
}

function collect_homophone_mapping() {
    var mapping = {};

    $("#homophone-mapping-list .homophone-mapping-row").each(function () {
        var term = $(this).find(".homophone-term").val().trim();
        var value = $(this).find(".homophone-value").val().trim();

        if (term && value) {
            mapping[term] = value;
        }
    });

    return mapping;
}

function save_homophone_mapping() {
    var mapping = collect_homophone_mapping();

    $.ajax({
        type: "POST",
        url: "/admin/set_homophone_mapping",
        data: JSON.stringify({ homophone_mapping: mapping }),
        contentType: "application/json",
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function () {
            homophoneMappingData = mapping;
            alert("谐音映射已保存");
        },
        error: function () {
            alert("保存谐音映射失败，请查看日志！");
        }
    });
}

function render_polyphonic_mapping_list(mapping) {
    var container = $("#polyphonic-mapping-list");
    container.empty();

    var entries = Object.entries(mapping || {});
    if (entries.length === 0) {
        container.append('<div class="text-muted">暂无多音字词条</div>');
        return;
    }

    entries.forEach(function (entry) {
        container.append(create_polyphonic_mapping_row(entry[0], entry[1]));
    });
}

function create_polyphonic_mapping_row(word, pinyinList) {
    var row = $(`
        <div class="polyphonic-mapping-row language-mapping-row">
            <input type="text" class="form-control polyphonic-word" placeholder="词语">
            <input type="text" class="form-control polyphonic-pinyin" placeholder="拼音（空格分隔）">
            <button type="button" class="btn btn-outline-danger polyphonic-remove">删除</button>
        </div>
    `);

    var pinyinText = Array.isArray(pinyinList) ? pinyinList.join(" ") : (pinyinList || "");
    row.find(".polyphonic-word").val(word || "");
    row.find(".polyphonic-pinyin").val(pinyinText);
    return row;
}

function parse_polyphonic_pinyin(text) {
    return String(text || "")
        .split(/[\s,，]+/)
        .map(function (item) { return item.trim(); })
        .filter(function (item) { return item.length > 0; });
}

function add_polyphonic_mapping_from_inputs() {
    var word = $("#polyphonic-mapping-word").val().trim();
    var pinyinText = $("#polyphonic-mapping-pinyin").val().trim();
    var pinyinList = parse_polyphonic_pinyin(pinyinText);

    if (!word || pinyinList.length === 0) {
        alert("请先填写词语和拼音。\n拼音支持空格或逗号分隔。");
        return;
    }

    $("#polyphonic-mapping-list .text-muted").remove();
    $("#polyphonic-mapping-list").append(create_polyphonic_mapping_row(word, pinyinList));
    $("#polyphonic-mapping-word").val("");
    $("#polyphonic-mapping-pinyin").val("");
}

function collect_polyphonic_mapping() {
    var mapping = {};

    $("#polyphonic-mapping-list .polyphonic-mapping-row").each(function () {
        var word = $(this).find(".polyphonic-word").val().trim();
        var pinyinText = $(this).find(".polyphonic-pinyin").val().trim();
        var pinyinList = parse_polyphonic_pinyin(pinyinText);

        if (word && pinyinList.length > 0) {
            mapping[word] = pinyinList;
        }
    });

    return mapping;
}

function save_polyphonic_mapping() {
    var mapping = collect_polyphonic_mapping();

    $.ajax({
        type: "POST",
        url: "/admin/set_polyphonic_mapping",
        data: JSON.stringify({ polyphonic_mapping: mapping }),
        contentType: "application/json",
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function () {
            polyphonicMappingData = mapping;
            alert("多音字词典已保存");
        },
        error: function (xhr) {
            var message = "保存多音字词典失败，请查看日志！";
            if (xhr && xhr.responseJSON && xhr.responseJSON.message) {
                message = "保存失败：" + xhr.responseJSON.message;
            }
            alert(message);
        }
    });
}
