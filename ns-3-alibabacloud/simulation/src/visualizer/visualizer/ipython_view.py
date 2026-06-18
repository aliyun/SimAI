"""
Backend to the console plugin.

@author: Eitan Isaacson
@organization: IBM Corporation
@copyright: Copyright (c) 2007 IBM Corporation
@license: BSD

All rights reserved. This program and the accompanying materials are made 
available under the terms of the BSD which accompanies this distribution, and 
is available at U{http://www.opensource.org/licenses/bsd-license.php}
"""
# this file is a modified version of source code from the Accerciser project
# http://live.gnome.org/accerciser

from __future__ import print_function
import gtk, gobject
import re
import sys
import os
from gi.repository import Pango
from StringIO import StringIO
import IPython

from pkg_resources import parse_version

try:
  import IPython
except ImportError:
  ##@ var IPython
  # 
  IPython = None

## IterableIPShell class
class IterableIPShell:
  ## @var IP
  #  IP
  ## @var iter_more
  #  iterate more
  ## @var history_level
  #  history level
  ## @var complete_sep
  #  separators
  ## @var prompt
  #  prompt
  ## @var header
  #  header
  ## @var config
  #  config
  ## @var user_ns
  #  user_ns
  ## @var old_stdout
  #  saved stdout
  ## @var old_stderr
  #  saved stderr
  ## @var system
  #  system 
  ## @var cfg
  # configuration
  ## @var colors
  #  colors
  ## @var raw_input_original
  #  original raw input
  ## @var stdin
  #  cin
  ## @var stdout
  #  cout
  ## @var stderr
  #  cerr
  ## @var raw_input
  #  raw input
  ## @var excepthook
  #  exception hook 
  ## Constructor
  def __init__(self,argv=None,user_ns=None,user_global_ns=None,
               cin=None, cout=None,cerr=None, input_func=None):
    """! Initializer
    
    @param self: this object
    @param argv: Command line options for IPython
    @param user_ns: User namespace.
    @param user_global_ns: User global namespace.
    @param cin: Console standard input.
    @param cout: Console standard output.
    @param cerr: Console standard error.
    @param input_func: Replacement for builtin raw_input()
    """
    io = IPython.utils.io
    if input_func:
      if parse_version(IPython.release.version) >= parse_version("1.2.1"):
        IPython.terminal.interactiveshell.raw_input_original = input_func
      else:
        IPython.frontend.terminal.interactiveshell.raw_input_original = input_func
    if cin:
      io.stdin = io.IOStream(cin)
    if cout:
      io.stdout = io.IOStream(cout)
    if cerr:
      io.stderr = io.IOStream(cerr)

    # This is to get rid of the blockage that occurs during 
    # IPython.Shell.InteractiveShell.user_setup()

    io.raw_input = lambda x: None

    os.environ['TERM'] = 'dumb'
    excepthook = sys.excepthook 

    from IPython.config.loader import Config
    cfg = Config()
    cfg.InteractiveShell.colors = "Linux"

    # InteractiveShell's __init__ overwrites io.stdout,io.stderr with
    # sys.stdout, sys.stderr, this makes sure they are right
    #
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.stdout.stream, io.stderr.stream

    # InteractiveShell inherits from SingletonConfigurable, so use instance()
    #
    if parse_version(IPython.release.version) >= parse_version("1.2.1"):
      self.IP = IPython.terminal.embed.InteractiveShellEmbed.instance(\
              config=cfg, user_ns=user_ns)
    else:
      self.IP = IPython.frontend.terminal.embed.InteractiveShellEmbed.instance(\
              config=cfg, user_ns=user_ns)

    sys.stdout, sys.stderr = old_stdout, old_stderr

    self.IP.system = lambda cmd: self.shell(self.IP.var_expand(cmd),
                                            header='IPython system call: ')
#                                            local_ns=user_ns)
                                            #global_ns=user_global_ns)
                                            #verbose=self.IP.rc.system_verbose)

    self.IP.raw_input = input_func
    sys.excepthook = excepthook
    self.iter_more = 0
    self.history_level = 0
    self.complete_sep =  re.compile('[\s\{\}\[\]\(\)]')
    self.updateNamespace({'exit':lambda:None})
    self.updateNamespace({'quit':lambda:None})
    self.IP.readline_startup_hook(self.IP.pre_readline)
    # Workaround for updating namespace with sys.modules
    #
    self.__update_namespace()

  def __update_namespace(self):
    """!
    Update self.IP namespace for autocompletion with sys.modules
    @return none
    """
    for k, v in list(sys.modules.items()):
        if not '.' in k:
          self.IP.user_ns.update({k:v})

  def execute(self):
    """!
    Executes the current line provided by the shell object.
    @return none
    """
    self.history_level = 0
    orig_stdout = sys.stdout
    sys.stdout = IPython.utils.io.stdout

    orig_stdin = sys.stdin
    sys.stdin = IPython.utils.io.stdin;
    self.prompt = self.generatePrompt(self.iter_more)

    self.IP.hooks.pre_prompt_hook()
    if self.iter_more:
        try:
            self.prompt = self.generatePrompt(True)
        except:
            self.IP.showtraceback()
        if self.IP.autoindent:
            self.IP.rl_do_indent = True

    try:
      line = self.IP.raw_input(self.prompt)
    except KeyboardInterrupt:
      self.IP.write('\nKeyboardInterrupt\n')
      self.IP.input_splitter.reset()
    except:
      self.IP.showtraceback()
    else:
      self.IP.input_splitter.push(line)
      self.iter_more = self.IP.input_splitter.push_accepts_more()
      self.prompt = self.generatePrompt(self.iter_more)
      if (self.IP.SyntaxTB.last_syntax_error and
          self.IP.autoedit_syntax):
          self.IP.edit_syntax_error()
      if not self.iter_more:
          if parse_version(IPython.release.version) >= parse_version("2.0.0-dev"):
            source_raw = self.IP.input_splitter.raw_reset()
          else:
            source_raw = self.IP.input_splitter.source_raw_reset()[1]
          self.IP.run_cell(source_raw, store_history=True)
          self.IP.rl_do_indent = False
      else:
          # TODO: Auto-indent
          #
          self.IP.rl_do_indent = True
          pass

    sys.stdout = orig_stdout
    sys.stdin = orig_stdin

  def generatePrompt(self, is_continuation):
    """!
    Generate prompt depending on is_continuation value

    @param is_continuation
    @return: The prompt string representation

    """

    # Backwards compatibility with ipyton-0.11
    #
    ver = IPython.__version__
    if '0.11' in ver:
        prompt = self.IP.hooks.generate_prompt(is_continuation)
    else:
        if is_continuation:
            prompt = self.IP.prompt_manager.render('in2')
        else:
            prompt = self.IP.prompt_manager.render('in')

    return prompt


  def historyBack(self):
    """!
    Provides one history command back.
    
    @param self this object
    @return: The command string.
    """
    self.history_level -= 1
    if not self._getHistory():
      self.history_level +=1
    return self._getHistory()

  def historyForward(self):
    """!
    Provides one history command forward.
    
    @param self this object
    @return: The command string.
    """
    if self.history_level < 0:
      self.history_level += 1
    return self._getHistory()

  def _getHistory(self):
    """!
    Gets the command string of the current history level.
    
    @param self this object
    @return: Historic command string.
    """
    try:
      rv = self.IP.user_ns['In'][self.history_level].strip('\n')
    except IndexError:
      rv = ''
    return rv

  def updateNamespace(self, ns_dict):
    """!
    Add the current dictionary to the shell namespace.
    
    @param ns_dict: A dictionary of symbol-values.
    @return none
    """
    self.IP.user_ns.update(ns_dict)

  def complete(self, line):
    """!
    Returns an auto completed line and/or possibilities for completion.
    
    @param line: Given line so far.
    @return: Line completed as for as possible, and possible further completions.
    """
    split_line = self.complete_sep.split(line)
    if split_line[-1]:
      possibilities = self.IP.complete(split_line[-1])
    else:
      completed = line
      possibilities = ['', []]
    if possibilities:
      def _commonPrefix(str1, str2):
        """!
        Reduction function. returns common prefix of two given strings.
        
        @param str1: First string.
        @param str2: Second string
        @return: Common prefix to both strings.
        """
        for i in range(len(str1)):
          if not str2.startswith(str1[:i+1]):
            return str1[:i]
        return str1
      if possibilities[1]:
        common_prefix = reduce(_commonPrefix, possibilities[1]) or line[-1]
        completed = line[:-len(split_line[-1])]+common_prefix
      else:
        completed = line
    else:
      completed = line
    return completed, possibilities[1]
  

  def shell(self, cmd,verbose=0,debug=0,header=''):
    """!
    Replacement method to allow shell commands without them blocking.
    
    @param cmd: Shell command to execute.
    @param verbose: Verbosity
    @param debug: Debug level
    @param header: Header to be printed before output
    @return none
    """
    stat = 0
    if verbose or debug: print(header+cmd)
    # flush stdout so we don't mangle python's buffering
    if not debug:
      input, output = os.popen4(cmd)
      print(output.read())
      output.close()
      input.close()

## ConsoleView class
class ConsoleView(Gtk.TextView):
  ## @var ANSI_COLORS
  #  color list
  ## @var text_buffer
  #  text buffer
  ## @var mark
  #  scroll mark
  ## @var color_pat
  #  color pattern
  ## @var line_start
  #  line start
  """
  Specialized text view for console-like workflow.

  @cvar ANSI_COLORS: Mapping of terminal colors to X11 names.
  @type ANSI_COLORS: dictionary

  @ivar text_buffer: Widget's text buffer.
  @type text_buffer: Gtk.TextBuffer
  @ivar color_pat: Regex of terminal color pattern
  @type color_pat: _sre.SRE_Pattern
  @ivar mark: Scroll mark for automatic scrolling on input.
  @type mark: Gtk.TextMark
  @ivar line_start: Start of command line mark.
  @type line_start: Gtk.TextMark
  """
  ANSI_COLORS =  {'0;30': 'Black',     '0;31': 'Red',
                  '0;32': 'Green',     '0;33': 'Brown',
                  '0;34': 'Blue',      '0;35': 'Purple',
                  '0;36': 'Cyan',      '0;37': 'LightGray',
                  '1;30': 'DarkGray',  '1;31': 'DarkRed',
                  '1;32': 'SeaGreen',  '1;33': 'Yellow',
                  '1;34': 'LightBlue', '1;35': 'MediumPurple',
                  '1;36': 'LightCyan', '1;37': 'White'}

  def __init__(self):
    """
    Initialize console view.
    """
    GObject.GObject.__init__(self)
    self.modify_font(Pango.FontDescription('Mono'))
    self.set_cursor_visible(True)
    self.text_buffer = self.get_buffer()
    self.mark = self.text_buffer.create_mark('scroll_mark',
                                             self.text_buffer.get_end_iter(),
                                             False)
    for code in self.ANSI_COLORS:
      self.text_buffer.create_tag(code, 
                                  foreground=self.ANSI_COLORS[code], 
                                  weight=700)
    self.text_buffer.create_tag('0')
    self.text_buffer.create_tag('notouch', editable=False)
    self.color_pat = re.compile('\x01?\x1b\[(.*?)m\x02?')
    self.line_start = \
        self.text_buffer.create_mark('line_start', 
                                     self.text_buffer.get_end_iter(), True)
    self.connect('key-press-event', self.onKeyPress)
    
  def write(self, text, editable=False):
    """!
    Write given text to buffer.
    
    @param text: Text to append.
    @param editable: If true, added text is editable.
    @return none
    """
    GObject.idle_add(self._write, text, editable)

  def _write(self, text, editable=False):
    """!
    Write given text to buffer.
    
    @param text: Text to append.
    @param editable: If true, added text is editable.
    @return none
    """
    segments = self.color_pat.split(text)
    segment = segments.pop(0)
    start_mark = self.text_buffer.create_mark(None,
                                              self.text_buffer.get_end_iter(),
                                              True)
    self.text_buffer.insert(self.text_buffer.get_end_iter(), segment)

    if segments:
      ansi_tags = self.color_pat.findall(text)
      for tag in ansi_tags:
        i = segments.index(tag)
        self.text_buffer.insert_with_tags_by_name(self.text_buffer.get_end_iter(),
                                                  segments[i+1], str(tag))
        segments.pop(i)
    if not editable:
      self.text_buffer.apply_tag_by_name('notouch',
                                         self.text_buffer.get_iter_at_mark(start_mark),
                                         self.text_buffer.get_end_iter())
    self.text_buffer.delete_mark(start_mark)
    self.scroll_mark_onscreen(self.mark)

  def showPrompt(self, prompt):
    """!
    Prints prompt at start of line.
    
    @param prompt: Prompt to print.
    @return none
    """
    GObject.idle_add(self._showPrompt, prompt)

  def _showPrompt(self, prompt):
    """!
    Prints prompt at start of line.
    
    @param prompt: Prompt to print.
    @return none
    """
    self._write(prompt)
    self.text_buffer.move_mark(self.line_start,
                               self.text_buffer.get_end_iter())

  def changeLine(self, text):
    """!
    Replace currently entered command line with given text.
    
    @param text: Text to use as replacement.
    @return none
    """
    GObject.idle_add(self._changeLine, text)

  def _changeLine(self, text):
    """!
    Replace currently entered command line with given text.
    
    @param text: Text to use as replacement.
    @return none
    """
    iter = self.text_buffer.get_iter_at_mark(self.line_start)
    iter.forward_to_line_end()
    self.text_buffer.delete(self.text_buffer.get_iter_at_mark(self.line_start), iter)
    self._write(text, True)

  def getCurrentLine(self):
    """!
    Get text in current command line.
    
    @return Text of current command line.
    """
    rv = self.text_buffer.get_slice(
      self.text_buffer.get_iter_at_mark(self.line_start),
      self.text_buffer.get_end_iter(), False)
    return rv

  def showReturned(self, text):
    """!
    Show returned text from last command and print new prompt.
    
    @param text: Text to show.
    @return none
    """
    GObject.idle_add(self._showReturned, text)

  def _showReturned(self, text):
    """!
    Show returned text from last command and print new prompt.
    
    @param text: Text to show.
    @return none
    """
    iter = self.text_buffer.get_iter_at_mark(self.line_start)
    iter.forward_to_line_end()
    self.text_buffer.apply_tag_by_name(
      'notouch', 
      self.text_buffer.get_iter_at_mark(self.line_start),
      iter)
    self._write('\n'+text)
    if text:
      self._write('\n')
    self._showPrompt(self.prompt)
    self.text_buffer.move_mark(self.line_start,self.text_buffer.get_end_iter())
    self.text_buffer.place_cursor(self.text_buffer.get_end_iter())

    if self.IP.rl_do_indent:
      indentation = self.IP.input_splitter.indent_spaces * ' '
      self.text_buffer.insert_at_cursor(indentation)

  def onKeyPress(self, widget, event):
    """!
    Key press callback used for correcting behavior for console-like 
    interfaces. For example 'home' should go to prompt, not to beginning of
    line.
    
    @param widget: Widget that key press accored in.
    @param event: Event object
    @return Return True if event should not trickle.
    """
    insert_mark = self.text_buffer.get_insert()
    insert_iter = self.text_buffer.get_iter_at_mark(insert_mark)
    selection_mark = self.text_buffer.get_selection_bound()
    selection_iter = self.text_buffer.get_iter_at_mark(selection_mark)
    start_iter = self.text_buffer.get_iter_at_mark(self.line_start)
    if event.keyval == Gdk.KEY_Home:
      if event.get_state() & Gdk.ModifierType.CONTROL_MASK or event.get_state() & Gdk.ModifierType.MOD1_MASK:
        pass
      elif event.get_state() & Gdk.ModifierType.SHIFT_MASK:
        self.text_buffer.move_mark(insert_mark, start_iter)
        return True
      else:
        self.text_buffer.place_cursor(start_iter)
        return True
    elif event.keyval == Gdk.KEY_Left:
      insert_iter.backward_cursor_position()
      if not insert_iter.editable(True):
        return True
    elif not event.string:
      pass
    elif start_iter.compare(insert_iter) <= 0 and \
          start_iter.compare(selection_iter) <= 0:
      pass
    elif start_iter.compare(insert_iter) > 0 and \
          start_iter.compare(selection_iter) > 0:
      self.text_buffer.place_cursor(start_iter)
    elif insert_iter.compare(selection_iter) < 0:
      self.text_buffer.move_mark(insert_mark, start_iter)
    elif insert_iter.compare(selection_iter) > 0:
      self.text_buffer.move_mark(selection_mark, start_iter)             

    return self.onKeyPressExtend(event)

  def onKeyPressExtend(self, event):
    """!
    For some reason we can't extend onKeyPress directly (bug #500900).
    @param event key press 
    @return none
    """
    pass

## IPythonView class
class IPythonView(ConsoleView, IterableIPShell):
  ## @var cout
  #  cout
  ## @var interrupt
  #  interrupt
  ## @var execute
  #  execute
  ## @var prompt
  #  prompt
  ## @var showPrompt
  #  show prompt
  ## @var history_pos
  #  history list
  ## @var window
  #  GTK Window
  """
  Sub-class of both modified IPython shell and L{ConsoleView} this makes
  a GTK+ IPython console.
  """
  def __init__(self):
    """
    Initialize. Redirect I/O to console.
    """
    ConsoleView.__init__(self)
    self.cout = StringIO()
    IterableIPShell.__init__(self, cout=self.cout,cerr=self.cout,
                             input_func=self.raw_input)
    self.interrupt = False
    self.execute()
    self.prompt = self.generatePrompt(False)
    self.cout.truncate(0)
    self.showPrompt(self.prompt)

  def raw_input(self, prompt=''):
    """!
    Custom raw_input() replacement. Gets current line from console buffer.
    
    @param prompt: Prompt to print. Here for compatibility as replacement.
    @return The current command line text.
    """
    if self.interrupt:
      self.interrupt = False
      raise KeyboardInterrupt
    return self.getCurrentLine()

  def onKeyPressExtend(self, event):
    """!
    Key press callback with plenty of shell goodness, like history, 
    autocompletions, etc.
    
    @param event: Event object.
    @return True if event should not trickle.
    """

    if event.get_state() & Gdk.ModifierType.CONTROL_MASK and event.keyval == 99:
      self.interrupt = True
      self._processLine()
      return True
    elif event.keyval == Gdk.KEY_Return:
      self._processLine()
      return True
    elif event.keyval == Gdk.KEY_Up:
      self.changeLine(self.historyBack())
      return True
    elif event.keyval == Gdk.KEY_Down:
      self.changeLine(self.historyForward())
      return True
    elif event.keyval == Gdk.KEY_Tab:
      if not self.getCurrentLine().strip():
        return False
      completed, possibilities = self.complete(self.getCurrentLine())
      if len(possibilities) > 1:
        slice = self.getCurrentLine()
        self.write('\n')
        for symbol in possibilities:
          self.write(symbol+'\n')
        self.showPrompt(self.prompt)
      self.changeLine(completed or slice)
      return True

  def _processLine(self):
    """!
    Process current command line.
    @return none
    """
    self.history_pos = 0
    self.execute()
    rv = self.cout.getvalue()
    if rv: rv = rv.strip('\n')
    self.showReturned(rv)
    self.cout.truncate(0)
    self.cout.seek(0)
 
if __name__ == "__main__":
  window = Gtk.Window()
  window.set_default_size(640, 320)
  window.connect('delete-event', lambda x, y: Gtk.main_quit())
  window.add(IPythonView())
  window.show_all()
  Gtk.main()
    
