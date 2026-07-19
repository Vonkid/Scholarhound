(function() {
  "use strict";

  var logicBeliefMap = null;
  var logicModel = null;

  function text(value) {
    return value == null ? "" : String(value);
  }

  function html(value) {
    return text(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function words(value) {
    return text(value)
      .toLowerCase()
      .replace(/bioelectronics/g, "bioelectronic")
      .replace(/electrochemical/g, "electronic")
      .split(/[^a-z0-9]+/)
      .filter(function(word) { return word.length >= 3; });
  }

  function unique(items) {
    var seen = {};
    return items.filter(function(item) {
      var key = text(item);
      if (!key || seen[key]) return false;
      seen[key] = true;
      return true;
    });
  }

  function countEvidence(belief, relation) {
    return (((belief || {}).evidence || {})[relation] || []).length;
  }

  function beliefState(belief) {
    var unresolved = countEvidence(belief, "contest") + countEvidence(belief, "pending");
    if (unresolved) return {key: "contested", label: "contested"};
    if (countEvidence(belief, "challenge")) return {key: "pressure", label: "under pressure"};
    if (countEvidence(belief, "support")) return {key: "formalised", label: "formalised"};
    return {key: "thin", label: "thin evidence"};
  }

  function storyNode(group, type) {
    return ((group || {}).nodes || []).find(function(node) {
      return node.node_type === type;
    }) || null;
  }

  function beliefMatchScore(group, belief) {
    var direction = words((group || {}).direction).join(" ");
    var domain = text((belief || {}).domain).toLowerCase();
    if (domain.indexOf("research-os") >= 0) return 0;
    var score = 0;
    var concepts = (belief.linked_concepts || []).map(function(item) {
      return words(item).join(" ");
    });
    concepts.forEach(function(concept) {
      if (concept && direction.indexOf(concept) >= 0) score += 3;
    });
    if (direction.indexOf("bioelectronic") >= 0 && concepts.indexOf("oect") >= 0) score += 3;
    if (direction.indexOf("organoid") >= 0 && concepts.indexOf("organoid") >= 0) score += 3;
    if (direction.indexOf("nanophotonic") >= 0 && concepts.indexOf("nanophotonic") >= 0) score += 3;
    var directionWords = unique(words((group || {}).direction));
    var beliefWords = unique(words((belief || {}).title).concat(words((belief || {}).claim)));
    directionWords.forEach(function(word) {
      if (beliefWords.indexOf(word) >= 0) score += .25;
    });
    return score;
  }

  function matchingBeliefs(group) {
    var beliefs = (logicBeliefMap || {}).beliefs || [];
    return beliefs
      .map(function(belief) { return {belief: belief, score: beliefMatchScore(group, belief)}; })
      .filter(function(item) { return item.score >= 4; })
      .sort(function(a, b) { return b.score - a.score; })
      .slice(0, 3)
      .map(function(item) { return item.belief; });
  }

  function formalBeliefNode(belief) {
    var state = beliefState(belief);
    return {
      id: "logic-belief-" + belief.id,
      kind: "belief",
      className: state.key,
      kicker: "Belief / " + (belief.display_id || "V3"),
      title: belief.title || belief.claim,
      state: state.label,
      count: countEvidence(belief, "support") + countEvidence(belief, "challenge") + countEvidence(belief, "contest") + countEvidence(belief, "pending"),
      belief: belief
    };
  }

  function relationNode(belief, relation, label, className) {
    var relationMap = {
      support: ["support"],
      pressure: ["challenge"],
      unresolved: ["contest", "pending"]
    };
    var count = relationMap[relation].reduce(function(total, key) {
      return total + countEvidence(belief, key);
    }, 0);
    var titles = {
      support: "Evidence in the same direction",
      pressure: "Counter-evidence and scope pressure",
      unresolved: "Contested or underdetermined boundary"
    };
    return {
      id: "logic-relation-" + relation + "-" + belief.id,
      kind: "relation",
      className: className,
      kicker: label,
      title: titles[relation],
      state: count ? label : "no recorded items",
      count: count,
      relation: relation,
      belief: belief
    };
  }

  function informalNode(node, kind) {
    return {
      id: "logic-story-" + (node ? node.id : kind),
      kind: kind || "story",
      className: kind === "next" ? "next-question" : "not-formalised",
      kicker: node ? (node.type_label || "Trajectory state") : "Trajectory state",
      title: node ? node.title : "No structured state recorded",
      state: kind === "next" ? "open question" : "not yet formalised",
      count: node ? Number(node.evidence_count || 0) : 0,
      story: node
    };
  }

  function buildLogicModel(group) {
    var question = storyNode(group, "question") || ((group || {}).nodes || [])[0] || null;
    var working = storyNode(group, "working_hypothesis");
    var next = storyNode(group, "next_question") || ((group || {}).nodes || []).slice(-1)[0] || null;
    var beliefs = matchingBeliefs(group);
    var root = {
      id: "logic-root-" + text((group || {}).id || "trajectory"),
      kind: "question",
      className: "question",
      kicker: "Core question",
      title: question ? question.title : text((group || {}).direction || "Research question"),
      state: "question",
      count: Number((group || {}).evidence_count || 0),
      story: question
    };
    var beliefNodes;
    var stateNodes;
    if (beliefs.length) {
      beliefNodes = beliefs.map(formalBeliefNode);
      var lead = beliefs[0];
      stateNodes = [
        relationNode(lead, "support", "support", "support"),
        relationNode(lead, "pressure", "pressure", "pressure"),
        relationNode(lead, "unresolved", "unresolved", "unresolved")
      ];
    } else {
      beliefNodes = [informalNode(working || question, "belief-draft")];
      stateNodes = [
        informalNode(storyNode(group, "turning_point"), "story"),
        informalNode(storyNode(group, "conceptual_shift"), "story"),
        informalNode(storyNode(group, "current_model"), "story")
      ];
    }
    var nextNode = informalNode(next, "next");
    var all = [root].concat(beliefNodes, stateNodes, [nextNode]);
    return {
      group: group,
      root: root,
      beliefs: beliefNodes,
      states: stateNodes,
      next: nextNode,
      all: all,
      formalised: beliefs.length > 0
    };
  }

  function ensureWorkspace() {
    var stage = document.querySelector("#tab-trajectory .trajectory-stage");
    var panel = document.getElementById("trajectory-panel");
    if (!stage || !panel) return;
    if (!stage.parentNode.classList.contains("logic-workspace")) {
      var workspace = document.createElement("div");
      workspace.className = "logic-workspace";
      stage.parentNode.insertBefore(workspace, stage);
      workspace.appendChild(stage);
      workspace.appendChild(panel);
    }
    var heading = document.querySelector("#tab-trajectory .trajectory-header .section-title");
    if (heading) heading.textContent = "Trajectory Logic";
    var legend = document.querySelector("#tab-trajectory .trajectory-header > div:last-child");
    if (legend) {
      legend.innerHTML =
        '<span class="badge badge-green">Formalised</span>' +
        '<span class="badge badge-amber">Under pressure</span>' +
        '<span class="badge badge-blue">Not formalised</span>';
    }
  }

  function nodeMarkup(node) {
    var selected = selectedTrajectoryId === node.id ? " selected" : "";
    var countLabel = "";
    if (node.kind === "question" || node.kind === "story" || node.kind === "belief-draft") {
      countLabel = node.count + " coverage";
    } else if (node.kind === "belief") {
      countLabel = node.count + " sources";
    } else if (node.kind !== "next") {
      countLabel = String(node.count);
    }
    return '<button type="button" class="logic-node ' + html(node.className) + selected + '" data-logic-node="' + html(node.id) + '">' +
      '<p class="logic-node-kicker">' + html(node.kicker) + '</p>' +
      '<p class="logic-node-title">' + html(node.title) + '</p>' +
      '<div class="logic-node-meta"><span class="logic-state"><span class="logic-state-dot"></span>' + html(node.state) + '</span>' +
        (countLabel ? '<span class="logic-count">' + html(countLabel) + '</span>' : '') + '</div>' +
    '</button>';
  }

  function renderLogicTree(container, model) {
    var group = model.group || {};
    if (!selectedTrajectoryId || !model.all.some(function(node) { return node.id === selectedTrajectoryId; })) {
      selectedTrajectoryId = (model.beliefs[0] || model.root).id;
    }
    var boundary = model.formalised
      ? "Direction and scope remain human-authorised. The kernel records support, pressure and unresolved evidence; it does not turn coverage into truth."
      : "No formal belief is attached to this direction. Coverage is visible, but no settled, contested or thin-evidence state is inferred."
    container.innerHTML =
      '<div class="logic-canvas">' +
        '<div class="logic-map-meta"><div><p class="logic-map-kicker">Question -> belief -> evidence -> next decision</p><h3 class="logic-map-title">' + html(group.direction || "Research trajectory") + '</h3></div><div class="logic-coverage">' + html(group.evidence_count || 0) + ' coverage points</div></div>' +
        '<div class="logic-tree" role="tree" aria-label="Trajectory logic">' +
          '<div class="logic-node-row single">' + nodeMarkup(model.root) + '</div>' +
          '<div class="logic-edge"></div><div class="logic-relation-label">frames</div>' +
          '<div class="logic-node-row branches' + (model.beliefs.length === 1 ? " one" : "") + '" style="--logic-columns:' + model.beliefs.length + '">' + model.beliefs.map(nodeMarkup).join("") + '</div>' +
          '<div class="logic-edge"></div><div class="logic-relation-label">tested by</div>' +
          '<div class="logic-node-row branches' + (model.states.length === 1 ? " one" : "") + '" style="--logic-columns:' + model.states.length + '">' + model.states.map(nodeMarkup).join("") + '</div>' +
          '<div class="logic-edge"></div><div class="logic-relation-label">opens</div>' +
          '<div class="logic-node-row single">' + nodeMarkup(model.next) + '</div>' +
        '</div>' +
        '<div class="logic-human-boundary"><strong>Human judgment boundary</strong>' + html(boundary) + '</div>' +
      '</div>';
    Array.prototype.forEach.call(container.querySelectorAll("[data-logic-node]"), function(button) {
      button.addEventListener("click", function() {
        window.selectTrajectoryNode(button.getAttribute("data-logic-node"));
      });
    });
  }

  function evidenceItems(belief, relation) {
    var evidence = (belief || {}).evidence || {};
    if (relation === "pressure") return evidence.challenge || [];
    if (relation === "unresolved") return (evidence.contest || []).concat(evidence.pending || []);
    return evidence.support || [];
  }

  function evidenceMarkup(items, relation) {
    if (!items.length) return '<div class="logic-empty">No evidence is recorded in this state.</div>';
    return '<div class="logic-evidence-list">' + items.slice(0, 5).map(function(item) {
      var source = item.source_ref || item.source_type || "source recorded";
      return '<article class="logic-evidence-item ' + html(relation) + '">' +
        '<strong>' + html(item.title || "Evidence") + '</strong>' +
        '<p>' + html(item.summary || "No summary recorded.") + '</p>' +
        '<div class="logic-evidence-meta">' + html(source) + ' / ' + html(item.strength || "strength not set") + '</div>' +
      '</article>';
    }).join("") + '</div>';
  }

  function beliefDetail(node) {
    var belief = node.belief;
    var state = beliefState(belief);
    var support = evidenceItems(belief, "support");
    var pressure = evidenceItems(belief, "pressure");
    var unresolved = evidenceItems(belief, "unresolved");
    var questions = belief.questions || [];
    var focus = node.kind === "relation" ? node.relation : "";
    var order = focus ? [focus].concat(["support", "pressure", "unresolved"].filter(function(item) { return item !== focus; })) : ["support", "pressure", "unresolved"];
    var labels = {support: "Supports current state", pressure: "Applies pressure", unresolved: "Unresolved boundary"};
    var sections = order.map(function(relation) {
      return '<section class="logic-section"><p class="logic-section-title">' + html(labels[relation]) + '</p>' + evidenceMarkup(evidenceItems(belief, relation), relation) + '</section>';
    }).join("");
    var review = questions.length
      ? '<div class="logic-review-box"><strong>Flagged for human judgment</strong>' + questions.map(function(question) { return '<div>' + html(question) + '</div>'; }).join("") + '</div>'
      : '<div class="logic-empty">No linked human question is recorded for this belief.</div>';
    return '<article class="logic-detail">' +
      '<header class="logic-detail-head"><p class="logic-detail-kicker">Belief / ' + html(belief.display_id || "V3") + '</p><h3 class="logic-detail-title">' + html(belief.title || belief.claim) + '</h3>' +
        '<div class="logic-facets"><span class="logic-facet formalised">' + html(state.label) + '</span><span class="logic-facet support">' + support.length + ' support</span><span class="logic-facet pressure">' + pressure.length + ' pressure</span><span class="logic-facet unresolved">' + unresolved.length + ' unresolved</span><span class="logic-facet">' + (belief.revision_history || []).length + ' revisions</span></div>' +
      '</header>' +
      '<div class="logic-detail-body"><p class="logic-semantics">' + html(belief.claim || "Literature state only.") + '</p>' + sections +
        '<section class="logic-section"><p class="logic-section-title">Human boundary</p>' + review + '</section>' +
      '</div>' +
    '</article>';
  }

  function paperMarkup(papers) {
    if (!papers.length) return '<div class="logic-empty">No supporting papers are attached to this trajectory node.</div>';
    return '<div class="logic-evidence-list">' + papers.slice(0, 5).map(function(paper) {
      return '<button type="button" class="logic-paper-button" data-logic-doi="' + html(paper.doi || "") + '">' + html(paper.title || "Untitled paper") + '<span>' + html(paper.journal || paper.doi || "source") + '</span></button>';
    }).join("") + '</div>';
  }

  function storyDetail(node) {
    var story = node.story || {};
    var isQuestion = node.kind === "question" || node.kind === "next";
    var boundary = node.kind === "next"
      ? '<div class="logic-review-box"><strong>Open research question</strong>' + html(story.missing_link || story.title || "No question recorded.") + '</div>'
      : '<div class="logic-empty">This node is coverage-level trajectory state. It has not been promoted into a V3 belief and carries no settled/contested judgment.</div>';
    return '<article class="logic-detail">' +
      '<header class="logic-detail-head"><p class="logic-detail-kicker">' + html(story.type_label || node.kicker) + '</p><h3 class="logic-detail-title">' + html(node.title) + '</h3>' +
        '<div class="logic-facets"><span class="logic-facet ' + (isQuestion ? "unresolved" : "") + '">' + html(node.state) + '</span><span class="logic-facet">' + html(node.count) + ' coverage points</span><span class="logic-facet">' + html(story.paper_count || (story.papers || []).length || 0) + ' papers</span></div>' +
      '</header>' +
      '<div class="logic-detail-body"><p class="logic-semantics">' + html(story.summary || "No structured summary is recorded.") + '</p>' +
        '<section class="logic-section"><p class="logic-section-title">State boundary</p>' + boundary + '</section>' +
        '<section class="logic-section"><p class="logic-section-title">Evidence coverage</p>' + paperMarkup(story.papers || []) + '</section>' +
        '<section class="logic-section"><p class="logic-section-title">Next move</p><div class="logic-empty">' + html(story.next_move || "No next action is recorded.") + '</div></section>' +
      '</div>' +
    '</article>';
  }

  function bindDetailActions(panel) {
    Array.prototype.forEach.call(panel.querySelectorAll("[data-logic-doi]"), function(button) {
      button.addEventListener("click", function() {
        var doi = button.getAttribute("data-logic-doi");
        if (doi && typeof window.openDbPaper === "function") window.openDbPaper(doi);
      });
    });
  }

  window.renderTrajectoryMapData = function(data) {
    trajectoryMapData = data;
    var groups = trajectoryMapData.story_groups || [];
    if (groups.length && !groups.find(function(group) { return group.id === selectedStoryId; })) {
      selectedStoryId = trajectoryMapData.story_direction_id || groups[0].id;
    }
    ensureWorkspace();
    window.renderTrajectorySwitcher();
    window.renderTrajectoryGraph();
    window.renderTrajectoryPanel();
  };

  window.renderTrajectorySwitcher = function() {
    var groups = (trajectoryMapData || {}).story_groups || [];
    document.getElementById("trajectory-info").textContent = (groups.length || 0) + " research directions";
    document.getElementById("trajectory-switcher").innerHTML = groups.map(function(group) {
      var active = group.id === selectedStoryId ? " active" : "";
      return '<button class="trajectory-switch' + active + '" data-logic-story="' + html(group.id) + '"><strong>' + html(group.direction) + '</strong><br><span>' + html(group.evidence_count || 0) + ' coverage points</span></button>';
    }).join("");
    Array.prototype.forEach.call(document.querySelectorAll("[data-logic-story]"), function(button) {
      button.addEventListener("click", function() {
        window.selectTrajectoryStory(button.getAttribute("data-logic-story"));
      });
    });
  };

  window.selectTrajectoryStory = function(id) {
    selectedStoryId = id;
    selectedTrajectoryId = "";
    window.renderTrajectorySwitcher();
    window.renderTrajectoryGraph();
    window.renderTrajectoryPanel();
  };

  window.selectTrajectoryNode = function(id) {
    selectedTrajectoryId = id;
    window.renderTrajectoryGraph();
    window.renderTrajectoryPanel();
  };

  window.renderTrajectoryGraph = function() {
    ensureWorkspace();
    var container = document.getElementById("trajectory-map");
    var group = typeof window.activeTrajectoryGroup === "function" ? window.activeTrajectoryGroup() : null;
    if (!group) {
      container.innerHTML = '<div class="empty">No trajectory evidence yet.</div>';
      return;
    }
    logicModel = buildLogicModel(group);
    renderLogicTree(container, logicModel);
  };

  window.renderTrajectoryPanel = function() {
    var panel = document.getElementById("trajectory-panel");
    if (!logicModel) {
      panel.innerHTML = "";
      return;
    }
    var selected = logicModel.all.find(function(node) { return node.id === selectedTrajectoryId; }) || logicModel.beliefs[0] || logicModel.root;
    panel.innerHTML = selected.belief ? beliefDetail(selected) : storyDetail(selected);
    bindDetailActions(panel);
  };

  function loadBeliefMap() {
    fetch("/api/kernel/v3/belief-map", {
      credentials: "same-origin",
      cache: "no-store",
      headers: {"Accept": "application/json"}
    }).then(function(response) {
      if (!response.ok) throw new Error("belief-map " + response.status);
      return response.json();
    }).then(function(data) {
      logicBeliefMap = data;
      if (trajectoryMapData && !document.getElementById("tab-trajectory").classList.contains("hidden")) {
        window.renderTrajectoryMapData(trajectoryMapData);
      }
    }).catch(function() {
      logicBeliefMap = {beliefs: [], review_requests: []};
    });
  }

  loadBeliefMap();
})();
