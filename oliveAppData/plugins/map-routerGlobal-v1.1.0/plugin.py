"""
Plugin/Pimiento used to do conditional processing  of LID scores into a ASR plugin/domain(s)
"""

import ast
import json
import operator
from olive.plugins import *
from distutils.util import strtobool


minimum_runtime_version = "5.1.0"
PLUGIN_MAPPING = "mapping"

import importlib
SCORE_THRESHOLD_KEY = 'global_threshold'
SELECT_BEST_KEY = 'select_best'

OPTION_DEFAULTS = {SCORE_THRESHOLD_KEY : 0.0, SELECT_BEST_KEY: True}

class CustomPlugin(Plugin, PluginToPlugin):

    def __init__(self):
        # Task name may need some improvement
        self.task = "P2P "
        # self.label = "ASR Plugin selection using LID (Global) Scores "
        #self.label = "Plugin selection using Global Scores "
        self.label = "MAP Conditional Workflow Router for Global Scorers"
        self.description = "Pimento used to select a plugin/domain based on global scores."
        self.vendor = "SRI"
        self.version = '1.1.0'
        self.minimum_runtime_version = "5.2.0"
        self.minimum_olive_version = "5.2.0"
        self.create_date = "2021-1-29"
        self.revision_date = "2021-1-29"
        self.group = "Olive"

        # values unique to this plugin:
        self.loaded_domains = []

        # This plugin is for internal use only, so it should NOT be advertised to clients
        self.protected = True


    def list_classes(self, domain_id):
        # do we have classes that we support?
        return []

    def load(self, domain_id):
        # Doesn't do much...  just parses the domain specific mapping files
        if domain_id not in self.loaded_domains:
            # Define domains
            domain = self.get_domains()[domain_id]
            # load the class id to plugin/domain mappings for for this plugin
            domain.map_config = dict()
            self._parse_mapping_config(domain)
            self.loaded_domains.append(domain_id)


    def run_plugin_analysis(self, domain_id, global_scores, workspace, classes=None, opts=None):
        # set or receive a score threshold... also need an option to pick the top score, top N scores, or all scores?
        threshold = OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]
        top_pick_only = OPTION_DEFAULTS[SELECT_BEST_KEY]

        # Check for any options:
        if opts is not None:
            threshold = float(opts[SCORE_THRESHOLD_KEY]) if SCORE_THRESHOLD_KEY in opts else threshold
            top_pick_only = bool(strtobool(opts[SELECT_BEST_KEY]) )if SELECT_BEST_KEY in opts \
                else top_pick_only

            logger.info("CLG asr select best: {}".format(top_pick_only))

        logger.info("CLG asr chooser options:, {}={}, {}={}".format(SCORE_THRESHOLD_KEY, threshold, SELECT_BEST_KEY, top_pick_only))

        domain = self.get_domains()[domain_id]
        if 'mapping' in opts:
            mapping_overrides = ast.literal_eval(opts['mapping'])
            for override in mapping_overrides:
                domain.map_config[PLUGIN_MAPPING][list(override.keys())[0]] = override[list(override.keys())[0]]

        # The domain should have a lookup table like:
        # lookup_table = {'eng': ('lid-asr-test', 'english'),
        #                 # 'eng': ('lid-asr-dynapy', 'eng-tdnnChain-tel-v1'),
        #                 'cmn': ('lid-asr-dynapy', 'cmn-tdnnChain-tel-v1'),
        #                 'fas': ('lid-asr-dynapy', 'fas-tdnn-tel-v1'),
        #                 'rus': ('lid-asr-dynapy', 'rus-tdnnChain-tel-v1'),
        #                 'spa': ('lid-asr-dynapy', 'spa-tdnnChain-tel-v1')}


        # global_scores is a WorkflowResult object since it was created by a GlobalScoring plugin:
        scores = global_scores

        # We don't assume global scores sorted, so sort 'em
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        logger.debug("Received global scores: {}".format(sorted_scores))

        plugin_domains = list()
        # set to true if one or more scores above the threshold could not be resolved into a plugin/domain
        no_plugin_for_high_score = False
        high_score_err_msg = None
        for gs in sorted_scores:
            if gs[1] > threshold:
                # add lang
                if gs[0] in domain.map_config[PLUGIN_MAPPING]:
                    pd = domain.map_config[PLUGIN_MAPPING][gs[0]]
                    plugin_domains.append({'plugin': pd[0], 'domain':pd[1]})
                else:
                    # Generate a warning
                    logger.warn("No plugin/domain available for class '{}' score({}).  Supported class names: {}."
                                .format(gs[0], gs[1], list(domain.map_config[PLUGIN_MAPPING].keys())))

                    if len(plugin_domains) == 0 and not no_plugin_for_high_score:
                        high_score_err_msg = "Unable to choose a plugin/domain since '{}' is not one of the supported class names: {}".format(gs[0],
                                                                      list(domain.map_config[PLUGIN_MAPPING].keys()))
                    no_plugin_for_high_score = True
                    # logger.error(err_msg_msg)
                    # self.escape_with_error(warn_msg)

        logger.debug("Final selected plugin domains: {}".format(plugin_domains))

        # validate there is a  score
        if len(plugin_domains) == 0:
            if high_score_err_msg:
                # this error message should let the client know that there was a high score(s) BUT there was no
                # supported plugin/domain for this class with the highest score
                self.escape_with_error(high_score_err_msg)
            else:
                self.escape_with_error("Score values too low to select a plugin/domain.")

        if top_pick_only:
            # only return the top score in our list
            return [plugin_domains[0]]

        else:
            # return all scores above threshold
            return plugin_domains


    def get_plugin_selection_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """

        trait_options = [
            TraitOption(SCORE_THRESHOLD_KEY, "Score Threshold",
                        "A Global scores must meet or exceed this threshold to select a plugin.",
                        TraitType.CHOICE_TRAIT, "", OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]),
            TraitOption(SELECT_BEST_KEY,
                        "Pick Top Score", "Select the best plugin/domain for the highest scoring result, otherwise "
                                          "return a selection for all scores that meet the score threshold",
                        TraitType.CHOICE_TRAIT, "", OPTION_DEFAULTS[SELECT_BEST_KEY]),
        ]
        return trait_options

    def _parse_mapping_config(self, domain):

        domain.map_config[PLUGIN_MAPPING] = dict()
        with open(os.path.join(domain.get_path(), 'mapping.json'), 'r', encoding="utf8") as f:
            raw_config = json.load(f)
            # print("plugin loaded config:", raw_config)

            for class_label in raw_config:
                # print("found:", class_label)
                if len(raw_config[class_label]) == 2:
                    domain.map_config[PLUGIN_MAPPING][class_label] = (raw_config[class_label][0], raw_config[class_label][1])
                else:
                    self.escape_with_error("Error parsing domain ({}) mapping configuration.  Expected a plugin and "
                                           "domain name but found '{}' for class: '{}'"
                                           .format(domain.get_id(), raw_config[class_label], class_label))


plugin = CustomPlugin()



