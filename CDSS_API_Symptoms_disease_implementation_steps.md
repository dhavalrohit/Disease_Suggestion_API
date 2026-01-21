## Following Changes to make in existing Code

### 1.Open application\views\admin\patient\appointment.php and add following lines:

#### Javascript Section:Add the following lines at the end after closing script (</script) tag OR can be added under exisiting Script Section:
```
<!-- Added By Raushan For Symptoms Selection Using DropDown -->

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.17.9/tagify.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.17.9/tagify.min.js"></script>

<script>
    var input = document.querySelector("#chiefcomplaint");

    var tagify = new Tagify(input, {
        whitelist: [],
        dropdown: {
            maxItems: 10,
            enabled: 1,
            closeOnSelect: false
        },
        addTagOnBlur: true,

// converts Tagify JSON tocomma-separated string

        originalInputValueFormat: valuesArr =>
            valuesArr.map(item => item.value).join(',')
    });

    tagify.on('input', function(e) {
        var value = e.detail.value;
        if (value.length < 1) return;

        fetch('<?php echo site_url("admin/symptoms/fetch_symptoms"); ?>?term=' + encodeURIComponent(value))
            .then(res => res.json())
            .then(function(data) {
                tagify.settings.whitelist = data;
                tagify.dropdown.show(value);
            });
    });
</script>
```

#### CSS/Style Sction:Add the following lines after existing closing style (/style) tag OR can be added under existing style section:
```
<style>
.tagify__tag {
        background: #3b82f6;
        color: #fff;
        border-radius: 4px;
        font-size: 13px;
    }

.tagify {
        min-height: 38px;
        height: auto !important;
        overflow: visible;
        border: 1px solid #d1d5db;
        padding: 4px;
    }
.chief-complaint-input {
        min-height: 38px;
        padding: 6px 10px;
        font-size: 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .chief-complaint-input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
    }

</style>
```
 