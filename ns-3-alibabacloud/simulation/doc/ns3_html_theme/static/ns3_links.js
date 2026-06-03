// nsnam.org links, independent of version
var ns3_site = "https://www.nsnam.org/";
var ns3_home = ns3_site + "";
var ns3_wiki = ns3_site + "wiki";
var ns3_bugs = "https://gitlab.com/nsnam/ns-3-dev/-/issues";
var ns3_merge = "https://gitlab.com/nsnam/ns-3-dev/-/merge_requests";

// Links that depend on version:
// Defined in ns3_version.js:

//   ns3_host    = "//"                      file://$PWD
//   ns3_version = "3.14"                    "3.14
//               = "ns-3-dev @ 12345abcde    "repo @ 12345abcde
//   ns3_release = "docs/release/3.14/"      "repo/"
//   ns3_local   = ""                        "build/"
//
// This lets us build several kinds of links:
//    http://www.nsnam.org/docs/manual/html/index.html
//    http://www.nsnam.org/docs/release/3.14/manual/html/index.html
//    file:///.../doc/manual/build/html/index.html
//
// with the appropriate version string.

var ns3_index = "index.html";
var ns3_rel  = ns3_host + ns3_release;
var ns3_api  = ns3_rel  + ns3_doxy          + ns3_index;

// The Sphinx layout template defines
//    var ns3_builder = "html", or "singlehtml"
// so we can condition the links on single/multi-page view.
if (typeof ns3_builder != 'undefined') {
  if (ns3_builder == "singlehtml") {
    ns3_index = ns3_local + "singlehtml/" + ns3_index;
  } else {
    ns3_index = ns3_local + "html/" + ns3_index;
  }
 }
 else {
   ns3_index = ns3_local + "html/" + ns3_index;
 }


var ns3_man  = ns3_rel  + "manual/"         + ns3_index;
var ns3_mod  = ns3_rel  + "models/"         + ns3_index;
var ns3_tut  = ns3_rel  + "tutorial/"       + ns3_index;
var ns3_con  = ns3_rel  + "contributing/"   + ns3_index;

function ns3_write_links() {
  document.getElementById("ns3_home1").href = ns3_home;
  document.getElementById("ns3_home2").href = ns3_home;
  document.getElementById("ns3_tut"  ).href = ns3_tut;
  document.getElementById("ns3_man"  ).href = ns3_man;
  document.getElementById("ns3_mod"  ).href = ns3_mod;
  document.getElementById("ns3_con"  ).href = ns3_con;
  document.getElementById("ns3_wiki" ).href = ns3_wiki;
  document.getElementById("ns3_api"  ).href = ns3_dox;
  document.getElementById("ns3_bugs" ).href = ns3_bugs;
  document.getElementById("ns3_merge").href = ns3_merge;
}
