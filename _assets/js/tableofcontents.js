/**
 * reveal.js table of contents plugin
 * 
 * A plugin which generates automatically a table of contents slide.
 * 
 * Demo https://naamor.github.io/reveal.js-tableofcontents/
 * 
 * MIT License
 * Copyright (c) 2018 Roman Stocker
 */

var RevealTableOfContents = window.RevealTableOfContents || (function () {
    // Set all option defaults
    var options = Reveal.getConfig().tableofcontents || {};
    var titleTag = options.titleTag || "h1";
    var titleTagSelector = ["h1", "h2", "h3", "h4", "h5", "h6"];
    var title = options.title || "Table of Contents";
    var position = options.position || 2;
    var fadeInElements = options.fadeInElements || false;

    var ignoreFirstSlide = options.ignoreFirstSlide;
    if (typeof ignoreFirstSlide === "undefined") ignoreFirstSlide = true;

    initialize();

    function initialize() {
        if (typeof options.titleTagSelector === "string") {
            titleTagSelector = options.titleTagSelector.split(",").map(item => {
                return item.trim();
            });
        }

        generateTableOfContentsSlide();
    }

    function generateTableOfContentsSlide() {
        var slides = document.getElementsByClassName("slides")[0];

        var section = document.createElement("section");

        var heading = document.createElement(titleTag);
        heading.innerText = title;
        section.appendChild(heading);

        var list = generateList();
        section.appendChild(list);

        // Subtract by one because index starts with zero
        var slideAfter = slides.children[position - 1];

        // Check if there are enough slides for the configured table of contents slide position
        // or set the table of contents slide automatically after the last slide
        if (slideAfter !== undefined) {
            slides.insertBefore(section, slideAfter);
        } else {
            slides.appendChild(section);
        }
    }

    // Generate list with the title of each slide
    function generateList() {
        var slides = Reveal.getSlides();

        var ul = document.createElement("ul");

        var counter = 0;

        // Ignore first slide with counter 0
        if (ignoreFirstSlide) {
            counter++;
        }

        for (counter; counter < slides.length; counter++) {
            var title = getTitle(slides[counter]);

            if (title !== undefined) {
                var li = document.createElement("li");

                // Add attributes for use reveal.js fragment functionality
                if (fadeInElements) {
                    li.className = "fragment";
                    li.setAttribute("data-fragment-index", counter);
                }

                li.innerText = title;

                ul.appendChild(li);
            }
        }

        return ul;
    }

    // Select the text of the most important heading tag of every slide
    function getTitle(slide) {
        return (title = Array.from(slide.childNodes)
            .filter(node => filterSlideTagElements(node))
            .sort((a, b) => sortHeadingTagElements(a, b))
            .map(node => node.textContent)[0]);
    }

    // Filter tags based on options
    function filterSlideTagElements(element) {
        if (element.tagName === undefined) {
            return false;
        }

        return titleTagSelector.indexOf(element.tagName.toLowerCase()) >= 0 && element.textContent !== "";
    }

    // Sort heading tags based on importance
    function sortHeadingTagElements(valueA, valueB) {
        if (valueA.tagName < valueB.tagName) return -1;
        if (valueA.tagName > valueB.tagName) return 1;
        return 0;
    }
})();
