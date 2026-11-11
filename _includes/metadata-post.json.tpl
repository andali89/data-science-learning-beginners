{% assign course_next = page.course_next | default: page.next %}
{
    "page": {
        "title": "Introduction",
        "level": "1.1",
        "depth": 1,
        {% if course_next %}
        "next": {
            "title": "{{course_next.title}}",
            "level": "1.2",
            "depth": 1,
            "path": "{{course_next.path | default: course_next.url}}",
            "ref": "{{course_next.path | default: course_next.url}}",
            "articles": []
        },
        {% endif %}
        "dir": "ltr"
    },

    {%- include metadata.json.tpl -%}
}
