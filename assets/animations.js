/*
 * Dashboard polish — runs in the browser; Dash auto-serves anything in
 * ``assets/``. No external libraries.
 *
 * Three behaviours:
 *   1. Scroll-triggered fade-up for every .card the first time it enters
 *      the viewport.
 *   2. KPI count-up: when the .kpi-value text changes to a new euro/percent
 *      number we tween between the old and new value over ~600ms.
 *   3. Refresh-button pulse: while the "prices-status" line is stale
 *      after a click on #refresh-prices-btn, the button shows an emerald
 *      halo. The halo stops the moment the status line refreshes.
 */
(function () {
    'use strict';

    // ----- 1. scroll fade-up ------------------------------------------------
    // Cards default to fully visible (no class). We only add ``fade-up-hidden``
    // if the card is currently below the fold AND JS is healthy — if anything
    // breaks, content stays visible.
    const FADE_SELECTOR = '.card';
    const fadeObs = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-up-visible');
                fadeObs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    function isBelowFold(el) {
        const rect = el.getBoundingClientRect();
        const vh = window.innerHeight || document.documentElement.clientHeight;
        // Treat anything whose top is >= 92% of the viewport as below the fold.
        return rect.top >= vh * 0.92;
    }

    function registerFadeTargets(root) {
        const nodes = (root === document ? document : root)
            .querySelectorAll(FADE_SELECTOR);
        nodes.forEach(function (el) {
            if (el.dataset.fadeRegistered === '1') return;
            el.dataset.fadeRegistered = '1';
            if (!isBelowFold(el)) return;   // already visible — leave alone
            el.classList.add('fade-up-hidden');
            fadeObs.observe(el);
        });
    }

    // Safety net: in case any card was hidden but the observer never fires
    // (browser quirk, focus visibility weirdness, etc.), reveal everything
    // after 2.5 s no matter what.
    setTimeout(function () {
        document.querySelectorAll('.fade-up-hidden:not(.fade-up-visible)')
            .forEach(function (el) {
                el.classList.add('fade-up-visible');
            });
    }, 2500);

    // ----- 2. KPI count-up --------------------------------------------------
    const NUMERIC_RE = /-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?/;

    function parseNumber(text) {
        if (text == null) return null;
        const m = text.match(NUMERIC_RE);
        if (!m) return null;
        return parseFloat(m[0].replace(/,/g, ''));
    }

    function formatLike(template, value) {
        // ``template`` is the new (target) string; we splice ``value`` into
        // its position so the prefix/suffix (e.g. "€", "%", "+") survive.
        const match = template.match(NUMERIC_RE);
        if (!match) return template;
        const decimals = (match[0].split('.')[1] || '').length || 2;
        const abs = Math.abs(value).toLocaleString('en-US', {
            minimumFractionDigits: decimals, maximumFractionDigits: decimals,
        });
        const signed = (value < 0 ? '-' : (match[0].startsWith('+') ? '+' : '')) + abs;
        return template.replace(NUMERIC_RE, signed);
    }

    const lastValues = new WeakMap();

    function animateNumber(el, finalText) {
        const target = parseNumber(finalText);
        if (target === null) {
            el.textContent = finalText;
            return;
        }
        const previous = lastValues.get(el);
        lastValues.set(el, target);

        // First render or unchanged value: skip the animation.
        if (previous === undefined || Math.abs(previous - target) < 0.005) {
            el.textContent = finalText;
            return;
        }

        // Keep the surrounding markup (the .kpi-value span sometimes
        // contains nested spans for color); only animate when it's a pure
        // text node we can rewrite safely.
        const start = previous;
        const startTime = performance.now();
        const duration = 600;
        function frame(now) {
            const t = Math.min(1, (now - startTime) / duration);
            // ease-out cubic
            const eased = 1 - Math.pow(1 - t, 3);
            const current = start + (target - start) * eased;
            el.textContent = formatLike(finalText, current);
            if (t < 1) requestAnimationFrame(frame);
            else el.textContent = finalText;
        }
        // Add the flash class for the brief visual cue.
        el.classList.remove('flash');
        // Force reflow so the animation restarts even if class was just removed.
        void el.offsetWidth;
        el.classList.add('flash');
        requestAnimationFrame(frame);
    }

    function findKpiValue(card) {
        return card.querySelector(':scope > .kpi-value');
    }

    function getDirectText(el) {
        // Concatenated direct text + immediate spans, so we get the headline
        // figure even if children are wrapped for color.
        let parts = [];
        el.childNodes.forEach(function (n) {
            if (n.nodeType === 3) parts.push(n.textContent);
            else if (n.nodeType === 1) parts.push(n.textContent);
        });
        return parts.join('').trim();
    }

    const kpiObserver = new MutationObserver(function (records) {
        const seen = new Set();
        records.forEach(function (r) {
            let card = r.target;
            while (card && !card.classList?.contains('kpi-card')) {
                card = card.parentNode;
            }
            if (!card || seen.has(card)) return;
            seen.add(card);
            const valueEl = findKpiValue(card);
            if (!valueEl) return;
            // Only animate if the kpi-value is a simple textual cell (no
            // span children that would clobber sub-formatting).
            if (valueEl.children.length === 0) {
                animateNumber(valueEl, valueEl.textContent.trim());
            } else {
                // Mixed children: just record the parsed value so we don't
                // double-animate on re-render but skip the tween.
                lastValues.set(valueEl, parseNumber(getDirectText(valueEl)));
                valueEl.classList.remove('flash');
                void valueEl.offsetWidth;
                valueEl.classList.add('flash');
            }
        });
    });

    function watchKpiCards() {
        const row = document.getElementById('kpi-cards');
        if (!row) return false;
        kpiObserver.observe(row, {
            childList: true, subtree: true, characterData: true,
        });
        return true;
    }

    // ----- 3. refresh-button pulse -----------------------------------------
    function watchRefreshPulse() {
        const btn = document.getElementById('refresh-prices-btn');
        const status = document.getElementById('prices-status');
        if (!btn || !status) return false;

        let lastStatus = status.textContent;
        btn.addEventListener('click', function () {
            btn.classList.add('is-pulsing');
            lastStatus = status.textContent;
        });

        new MutationObserver(function () {
            const cur = status.textContent;
            if (cur !== lastStatus && btn.classList.contains('is-pulsing')) {
                btn.classList.remove('is-pulsing');
                lastStatus = cur;
            }
        }).observe(status, {
            childList: true, characterData: true, subtree: true,
        });
        return true;
    }

    // ----- Boot ------------------------------------------------------------
    function boot() {
        registerFadeTargets(document);
        // Dash re-renders chunks of the DOM on every callback; watch for
        // newly inserted .card nodes so we keep animating them.
        const rootObs = new MutationObserver(function (records) {
            records.forEach(function (r) {
                r.addedNodes.forEach(function (n) {
                    if (n.nodeType === 1) registerFadeTargets(n);
                });
            });
        });
        rootObs.observe(document.body, { childList: true, subtree: true });

        // KPI row and refresh button may not exist on first paint — wait for
        // them.
        if (!watchKpiCards() || !watchRefreshPulse()) {
            const wait = new MutationObserver(function () {
                let done = true;
                if (!watchKpiCards()) done = false;
                if (!watchRefreshPulse()) done = false;
                if (done) wait.disconnect();
            });
            wait.observe(document.body, { childList: true, subtree: true });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
