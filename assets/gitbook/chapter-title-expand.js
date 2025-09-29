(function () {
    'use strict';

    var CHAPTER_SELECTOR = 'li.chapter-title';
    var ACTIVE_CHILD_SELECTOR = 'li.chapter.active';
    var TOGGLE_CLASS = 'expanded';
    var MANUAL_FLAG = 'chapterTitleManualExpanded';

    function hasActiveDescendant(element) {
        return !!(element && element.querySelector(ACTIVE_CHILD_SELECTOR));
    }

    function expand(element, options) {
        if (!element) {
            return;
        }

        if (options && options.manual) {
            element.dataset[MANUAL_FLAG] = 'true';
        }

        if (!element.classList.contains(TOGGLE_CLASS)) {
            element.classList.add(TOGGLE_CLASS);
        }
    }

    function collapse(element, options) {
        if (!element) {
            return;
        }

        if (options && options.manual) {
            delete element.dataset[MANUAL_FLAG];
        }

        if (element.classList.contains(TOGGLE_CLASS)) {
            element.classList.remove(TOGGLE_CLASS);
        }
    }

    function toggleChapterTitle(element) {
        if (!element) {
            return;
        }

        if (hasActiveDescendant(element)) {
            expand(element);
            return;
        }

        if (element.classList.contains(TOGGLE_CLASS)) {
            collapse(element, { manual: true });
        } else {
            expand(element, { manual: true });
        }
    }

    function syncChapterTitleState(element) {
        if (!element) {
            return;
        }

        if (hasActiveDescendant(element)) {
            expand(element);
        } else if (element.dataset[MANUAL_FLAG] === 'true') {
            expand(element);
        } else {
            collapse(element);
        }
    }

    function syncAllChapterTitles() {
        var chapterTitles = document.querySelectorAll(CHAPTER_SELECTOR);
        chapterTitles.forEach(function (element) {
            syncChapterTitleState(element);
        });
    }

    function setupMutationObservers() {
        var chapterTitles = document.querySelectorAll(CHAPTER_SELECTOR);
        if (!chapterTitles.length) {
            return;
        }

        var observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
                var target = mutation.target;
                var chapter = target.closest ? target.closest(CHAPTER_SELECTOR) : null;
                if (chapter) {
                    syncChapterTitleState(chapter);
                }
            });
        });

        chapterTitles.forEach(function (element) {
            observer.observe(element, {
                attributes: true,
                attributeFilter: ['class'],
                subtree: true,
                childList: true,
            });
        });
    }

    function setupToggleHandlers() {
        var chapterTitles = document.querySelectorAll(CHAPTER_SELECTOR);
        chapterTitles.forEach(function (element) {
            if (element.dataset.chapterTitleToggleBound === 'true') {
                return;
            }

            var trigger = element.querySelector('a');
            if (!trigger) {
                return;
            }

            trigger.addEventListener('click', function (event) {
                event.preventDefault();
                event.stopPropagation();
                toggleChapterTitle(element);
            });

            element.dataset.chapterTitleToggleBound = 'true';
        });
    }

    function initialize() {
        syncAllChapterTitles();
        setupMutationObservers();
        setupToggleHandlers();
    }

    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initialize);
        } else {
            initialize();
        }
    }

    if (typeof gitbook !== 'undefined' && gitbook.events) {
        gitbook.events.bind('page.change', function () {
            syncAllChapterTitles();
            setupToggleHandlers();
        });
    }
})();
