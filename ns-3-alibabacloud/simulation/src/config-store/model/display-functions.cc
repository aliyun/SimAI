/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "display-functions.h"
#include "raw-text-config.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/pointer.h"

namespace ns3 {
/*
 * This function includes the name of the attribute or the editable value
 * in the second column
 */
void
cell_data_function_col_1 (GtkTreeViewColumn *col, GtkCellRenderer *renderer,
                          GtkTreeModel *model, GtkTreeIter *iter, gpointer user_data)
{
  ModelNode *node = 0;
  gtk_tree_model_get (model, iter, COL_NODE, &node, -1);
  if (!node)
    {
      return;
    }
  if (node->type == ModelNode::NODE_ATTRIBUTE)
    {
      StringValue str;
      node->object->GetAttribute (node->name, str);
      g_object_set (renderer, "text", str.Get ().c_str (), (char*) 0);
      g_object_set (renderer, "editable", TRUE, (char*) 0);
    }
  else
    {
      g_object_set (renderer, "text", "", (char*) 0);
      g_object_set (renderer, "editable", FALSE, (char*) 0);
    }
}
/*
 * This function includes the name of the object, pointer, vector or vector item
 * in the first column
 */
void
cell_data_function_col_0 (GtkTreeViewColumn *col, GtkCellRenderer *renderer, GtkTreeModel *model,
                          GtkTreeIter *iter, gpointer user_data)
{
  ModelNode *node = 0;
  gtk_tree_model_get (model, iter, COL_NODE, &node, -1);
  g_object_set (renderer, "editable", FALSE, (char*) 0);
  if (!node)
    {
      return;
    }

  switch (node->type)
  {
    case ModelNode::NODE_OBJECT:
      g_object_set (renderer, "text", node->object->GetInstanceTypeId ().GetName ().c_str (), (char*) 0);
      break;
    case ModelNode::NODE_POINTER:
      g_object_set (renderer, "text", node->name.c_str (), (char*) 0);
      break;
    case ModelNode::NODE_VECTOR:
      g_object_set (renderer, "text", node->name.c_str (), (char*) 0);
      break;
    case ModelNode::NODE_VECTOR_ITEM:
      {
        std::stringstream oss;
        oss << node->index;
        g_object_set (renderer, "text", oss.str ().c_str (), (char*) 0);
      }
      break;
    case ModelNode::NODE_ATTRIBUTE:
      g_object_set (renderer, "text", node->name.c_str (), (char*) 0);
      break;
  }
}

/*
 * This is the callback called when the value of an attribute is changed
 */
void
cell_edited_callback (GtkCellRendererText *cell, gchar *path_string,
                      gchar *new_text, gpointer user_data)
{
  GtkTreeModel *model = GTK_TREE_MODEL (user_data);
  GtkTreeIter iter;
  gtk_tree_model_get_iter_from_string (model, &iter, path_string);
  ModelNode *node = 0;
  gtk_tree_model_get (model, &iter, COL_NODE, &node, -1);
  if (!node)
    {
      return;
    }
  NS_ASSERT (node->type == ModelNode::NODE_ATTRIBUTE);
  node->object->SetAttribute (node->name, StringValue (new_text));
}

/*
 * This function gets the column number 0 or 1 from the mouse
 * click
 */
int
get_col_number_from_tree_view_column (GtkTreeViewColumn *col)
{
  GList *cols;
  int num;
  g_return_val_if_fail (col != 0, -1);
  g_return_val_if_fail (gtk_tree_view_column_get_tree_view(col) != 0, -1);
  cols = gtk_tree_view_get_columns (GTK_TREE_VIEW (gtk_tree_view_column_get_tree_view(col)));
  num = g_list_index (cols, (gpointer) col);
  g_list_free (cols);
  return num;
}

/*
 * This function displays the tooltip for an object, pointer, vector
 * item or an attribute
 */
gboolean
cell_tooltip_callback (GtkWidget *widget, gint x, gint y, gboolean keyboard_tip,
                       GtkTooltip *tooltip, gpointer user_data)
{
  GtkTreeModel *model;
  GtkTreeIter iter;
  GtkTreeViewColumn * column;
  if (!gtk_tree_view_get_tooltip_context (GTK_TREE_VIEW (widget), &x, &y,
                                          keyboard_tip, &model, 0, &iter))
    {
      return FALSE;
    }
  if (!gtk_tree_view_get_path_at_pos (GTK_TREE_VIEW (widget), x, y, 0, &column, 0, 0))
    {
      return FALSE;
    }
  int col = get_col_number_from_tree_view_column (column);

  ModelNode *node = 0;
  gtk_tree_model_get (model, &iter, COL_NODE, &node, -1);
  if (!node)
    {
      return FALSE;
    }

  switch (node->type)
  {
    case ModelNode::NODE_OBJECT:
      if (col == 0)
        {
          std::string tip = "This object is of type "
              + node->object->GetInstanceTypeId ().GetName ();
          gtk_tooltip_set_text (tooltip, tip.c_str ());
          return TRUE;
        }
      break;
    case ModelNode::NODE_POINTER:
      if (col == 0)
        {
          PointerValue ptr;
          node->object->GetAttribute (node->name, ptr);
          std::string tip = "This object is of type "
              + ptr.GetObject ()->GetInstanceTypeId ().GetName ();
          gtk_tooltip_set_text (tooltip, tip.c_str ());
          return TRUE;
        }
      break;
    case ModelNode::NODE_VECTOR:
      break;
    case ModelNode::NODE_VECTOR_ITEM:
      if (col == 0)
        {
          std::string tip = "This object is of type "
              + node->object->GetInstanceTypeId ().GetName ();
          gtk_tooltip_set_text (tooltip, tip.c_str ());
          return TRUE;
        }
      break;
    case ModelNode::NODE_ATTRIBUTE:
      {
        uint32_t attrIndex = 0;
        TypeId tid;
        for (tid = node->object->GetInstanceTypeId (); tid.HasParent (); tid
        = tid.GetParent ())
          {
            for (uint32_t i = 0; i < tid.GetAttributeN (); ++i)
              {
                if (tid.GetAttribute (i).name == node->name)
                  {
                    attrIndex = i;
                    goto out;
                  }
              }
          }
        out: if (col == 0)
          {
            std::string tip = tid.GetAttribute (attrIndex).help;
            gtk_tooltip_set_text (tooltip, tip.c_str ());
          }
        else
          {
            struct TypeId::AttributeInformation info = tid.GetAttribute (attrIndex);
            Ptr<const AttributeChecker> checker = info.checker;
            std::string tip;
            tip = "This attribute is of type " + checker->GetValueTypeName ();
            if (checker->HasUnderlyingTypeInformation ())
              {
                tip += " " + checker->GetUnderlyingTypeInformation ();
              }
            gtk_tooltip_set_text (tooltip, tip.c_str ());
          }
        return TRUE;
      }
      break;
  }
  return FALSE;
}

/*
 * This is the main view opening the widget, getting tooltips and drawing the
 * tree of attributes...
 */
GtkWidget *
create_view (GtkTreeStore *model)
{
  GtkTreeViewColumn *col;
  GtkCellRenderer *renderer;
  GtkWidget *view;

  view = gtk_tree_view_new ();
  g_object_set (view, "has-tooltip", TRUE, (char*) 0);
  g_signal_connect (view, "query-tooltip", (GCallback) cell_tooltip_callback, 0);

  gtk_tree_view_set_grid_lines (GTK_TREE_VIEW (view), GTK_TREE_VIEW_GRID_LINES_BOTH);
  // gtk_tree_view_set_rules_hint (GTK_TREE_VIEW (view), TRUE);

  col = gtk_tree_view_column_new ();
  gtk_tree_view_column_set_title (col, "Object Attributes");
  gtk_tree_view_append_column (GTK_TREE_VIEW (view), col);
  renderer = gtk_cell_renderer_text_new ();
  gtk_tree_view_column_pack_start (col, renderer, TRUE);
  gtk_tree_view_column_set_cell_data_func (col, renderer, cell_data_function_col_0, 0, 0);
  g_object_set (renderer, "editable", FALSE, (char*) 0);

  col = gtk_tree_view_column_new ();
  gtk_tree_view_column_set_title (col, "Attribute Value");
  gtk_tree_view_append_column (GTK_TREE_VIEW (view), col);
  renderer = gtk_cell_renderer_text_new ();
  g_signal_connect (renderer, "edited", (GCallback) cell_edited_callback, model);
  gtk_tree_view_column_pack_start (col, renderer, TRUE);
  gtk_tree_view_column_set_cell_data_func (col, renderer, cell_data_function_col_1, 0, 0);

  gtk_tree_view_set_model (GTK_TREE_VIEW (view), GTK_TREE_MODEL (model));

  g_object_unref (model); /* destroy model automatically with view */

  return view;
}

/*
 * Exit the window when exit button is pressed
 */
void
exit_clicked_callback (GtkButton *button, gpointer user_data)
{
  gtk_main_quit ();
  gtk_widget_hide (GTK_WIDGET (user_data));
}

/*
 * Exit the application
 */
gboolean
delete_event_callback (GtkWidget *widget, GdkEvent *event, gpointer user_data)
{
  gtk_main_quit ();
  gtk_widget_hide (GTK_WIDGET (user_data));
  return TRUE;
}

/*
 * Delete the tree model contents
 */
gboolean
clean_model_callback (GtkTreeModel *model, GtkTreePath *path,
                      GtkTreeIter *iter, gpointer data)
{
  ModelNode *node = 0;
  gtk_tree_model_get (GTK_TREE_MODEL (model), iter, COL_NODE, &node, -1);
  if (node)
    {
      delete node;
    }
  gtk_tree_store_set (GTK_TREE_STORE (model), iter, COL_NODE, (ModelNode*) 0,
                      -1);
  return FALSE;
}

//     display functions used by default configurator
/*
 * This function writes data in the second column, this data is going to be editable
 * if it is a NODE_ATTRIBUTE
 */
void
cell_data_function_col_1_config_default (GtkTreeViewColumn *col, GtkCellRenderer *renderer,
                                         GtkTreeModel *model, GtkTreeIter *iter,
                                         gpointer user_data)
{
  ModelTypeid *node = 0;
  gtk_tree_model_get (model, iter, COL_TYPEID, &node, -1);
  if (!node)
    {
      return;
    }
  if (node->type == ModelTypeid::NODE_ATTRIBUTE)
    {
      g_object_set (renderer, "text", node->defaultValue.c_str (), (char*) 0);
      g_object_set (renderer, "editable", TRUE, (char*) 0);
    }
  else
    {
      g_object_set (renderer, "text", "", (char*) 0);
      g_object_set (renderer, "editable", FALSE, (char*) 0);
    }
}
/*
 * This function writes the attribute or typeid name in the column 0
 */
void
cell_data_function_col_0_config_default (GtkTreeViewColumn *col, GtkCellRenderer *renderer, GtkTreeModel *model,
                                         GtkTreeIter *iter, gpointer user_data)
{
  ModelTypeid *node = 0;
  gtk_tree_model_get (model, iter, COL_NODE, &node, -1);
  g_object_set (renderer, "editable", FALSE, (char*) 0);
  if (!node)
    {
      return;
    }

  switch (node->type)
  {
    case ModelTypeid::NODE_TYPEID:
      g_object_set (renderer, "text", node->tid.GetName ().c_str (), (char*) 0);
      break;
    case ModelTypeid::NODE_ATTRIBUTE:
      g_object_set (renderer, "text", node->name.c_str (), (char*) 0);
      break;
  }
}


/*
 *  This functions is called whenever there is a change in the value of an attribute
 *  If the input value is ok, it will be updated in the default value and in the
 *  gui, otherwise, it won't be updated in both.
 */
void
cell_edited_callback_config_default (GtkCellRendererText *cell, gchar *path_string,
                                     gchar *new_text, gpointer user_data)
{
  GtkTreeModel *model = GTK_TREE_MODEL (user_data);
  GtkTreeIter iter;
  gtk_tree_model_get_iter_from_string (model, &iter, path_string);
  ModelTypeid *node = 0;
  gtk_tree_model_get (model, &iter, COL_NODE, &node, -1);
  if (!node)
    {
      return;
    }
  NS_ASSERT (node->type == ModelTypeid::NODE_ATTRIBUTE);
  if (Config::SetDefaultFailSafe (node->tid.GetAttributeFullName (node->index),StringValue (new_text)))
    {
      node->defaultValue = new_text;
    }
}

/*
 * This function is used to display a tooltip whenever the user puts the mouse
 * over a type ID or an attribute. It will give the type and the possible values of
 * an attribute value and the type of the object for an attribute object or a
 * typeID object
 *
 * \param widget is the display object
 * \param x is the x position
 * \param y is the y position
 * \param keyboard_tip
 * \param tooltip is the tooltip information to be displayed
 * \param user_data
 * \return false if the tooltip is not displayed
 */
gboolean
cell_tooltip_callback_config_default (GtkWidget *widget, gint x, gint y,
                                      gboolean keyboard_tip, GtkTooltip *tooltip, gpointer user_data)
{
  GtkTreeModel *model;
  GtkTreeIter iter;
  GtkTreeViewColumn * column;
  if (!gtk_tree_view_get_tooltip_context (GTK_TREE_VIEW (widget), &x, &y,
                                          keyboard_tip, &model, 0, &iter))
    {
      return FALSE;
    }
  if (!gtk_tree_view_get_path_at_pos (GTK_TREE_VIEW (widget), x, y, 0, &column, 0, 0))
    {
      return FALSE;
    }
  int col = get_col_number_from_tree_view_column (column);

  ModelTypeid *node = 0;
  gtk_tree_model_get (model, &iter, COL_NODE, &node, -1);
  if (!node)
    {
      return FALSE;
    }

  switch (node->type)
  {
    case ModelTypeid::NODE_TYPEID:
      if (col == 0)
        {
          std::string tip = "This object is of type " + node->tid.GetName ();
          gtk_tooltip_set_text (tooltip, tip.c_str ());
          return TRUE;
        }
      break;
    case ModelTypeid::NODE_ATTRIBUTE:
      {
        uint32_t attrIndex = node->index;
        if (col == 0)
          {
            std::string tip = node->tid.GetAttribute (attrIndex).help;
            gtk_tooltip_set_text (tooltip, tip.c_str ());
          }
        else
          {
            Ptr<const AttributeChecker> checker = node->tid.GetAttribute (attrIndex).checker;
            std::string tip;
            tip = "This attribute is of type " + checker->GetValueTypeName ();
            if (checker->HasUnderlyingTypeInformation ())
              {
                tip += " " + checker->GetUnderlyingTypeInformation ();
              }
            gtk_tooltip_set_text (tooltip, tip.c_str ());
          }
        return TRUE;
      }
      break;
  }
  return FALSE;
}

/*
 * This is the action done when the user presses on the save button.
 * It will save the config to a file.
 *
 * \param button (unused)
 * \param user_data
 */
void
save_clicked_default (GtkButton *button, gpointer user_data)
{
  GtkWindow *parent_window = GTK_WINDOW (user_data);

  GtkFileChooserNative *native;
  GtkFileChooser *chooser;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_SAVE;
  gint res;

  native = gtk_file_chooser_native_new ("Save File",
                                        parent_window,
                                        action,
                                        "_Save",
                                        "_Cancel");
  chooser = GTK_FILE_CHOOSER (native);

  gtk_file_chooser_set_do_overwrite_confirmation (chooser, TRUE);

  gtk_file_chooser_set_current_name (chooser, ("config-defaults.txt"));

  res = gtk_native_dialog_run (GTK_NATIVE_DIALOG (native));
  if (res == GTK_RESPONSE_ACCEPT)
    {
      char *filename;

      filename = gtk_file_chooser_get_filename (chooser);
      RawTextConfigSave config;
      config.SetFilename (filename);
      config.Default ();
      g_free (filename);
    }

  g_object_unref (native);
}

/*
 * If the user presses the button load, it will load the config file into memory.
 *
 * \param button (unused)
 * \param user_data
 */
void
load_clicked_default (GtkButton *button, gpointer user_data)
{
  GtkWindow *parent_window = GTK_WINDOW (user_data);
  GtkFileChooserNative *native;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
  gint res;

  native = gtk_file_chooser_native_new ("Open File",
                                        parent_window,
                                        action,
                                        "_Open",
                                        "_Cancel");

  res = gtk_native_dialog_run (GTK_NATIVE_DIALOG (native));
  if (res == GTK_RESPONSE_ACCEPT)
    {
      char *filename;
      GtkFileChooser *chooser = GTK_FILE_CHOOSER (native);
      filename = gtk_file_chooser_get_filename (chooser);
      RawTextConfigLoad config;
      config.SetFilename (filename);
      config.Default ();
      g_free (filename);
    }

  g_object_unref (native);
}

/*
 * This is the action done when the user presses on the save button.
 * It will save the config to a file.
 *
 * \param button (unused)
 * \param user_data
 */
void
save_clicked_attribute (GtkButton *button, gpointer user_data)
{
  GtkWindow *parent_window = GTK_WINDOW (user_data);

  GtkFileChooserNative *native;
  GtkFileChooser *chooser;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_SAVE;
  gint res;

  native = gtk_file_chooser_native_new ("Save File",
                                        parent_window,
                                        action,
                                        "_Save",
                                        "_Cancel");
  chooser = GTK_FILE_CHOOSER (native);

  gtk_file_chooser_set_do_overwrite_confirmation (chooser, TRUE);

  gtk_file_chooser_set_current_name (chooser, ("config-attributes.txt"));

  res = gtk_native_dialog_run (GTK_NATIVE_DIALOG (native));
  if (res == GTK_RESPONSE_ACCEPT)
    {
      char *filename;

      filename = gtk_file_chooser_get_filename (chooser);
      RawTextConfigSave config;
      config.SetFilename (filename);
      config.Attributes ();
      g_free (filename);
    }

  g_object_unref (native);
}

/*
 * If the user presses the button load, it will load the config file into memory.
 *
 * \param button (unused)
 * \param user_data
 */
void
load_clicked_attribute (GtkButton *button, gpointer user_data)
{
  GtkWindow *parent_window = GTK_WINDOW (user_data);
  GtkFileChooserNative *native;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
  gint res;

  native = gtk_file_chooser_native_new ("Open File",
                                        parent_window,
                                        action,
                                        "_Open",
                                        "_Cancel");

  res = gtk_native_dialog_run (GTK_NATIVE_DIALOG (native));
  if (res == GTK_RESPONSE_ACCEPT)
    {
      char *filename;
      GtkFileChooser *chooser = GTK_FILE_CHOOSER (native);
      filename = gtk_file_chooser_get_filename (chooser);
      RawTextConfigLoad config;
      config.SetFilename (filename);
      config.Attributes ();
      g_free (filename);
    }

  g_object_unref (native);
}

/*
 * This is the main view opening the widget, getting tooltips and drawing the
 * tree of attributes
 */
GtkWidget *
create_view_config_default (GtkTreeStore *model)
{
  GtkTreeViewColumn *col;
  GtkCellRenderer *renderer;
  GtkWidget *view;

  view = gtk_tree_view_new ();
  g_object_set (view, "has-tooltip", TRUE, (char*) 0);
  g_signal_connect (view, "query-tooltip", (GCallback) cell_tooltip_callback_config_default, 0);

  gtk_tree_view_set_grid_lines (GTK_TREE_VIEW (view), GTK_TREE_VIEW_GRID_LINES_BOTH);
  // gtk_tree_view_set_rules_hint (GTK_TREE_VIEW (view), TRUE);

  col = gtk_tree_view_column_new ();
  gtk_tree_view_column_set_title (col, "Object Attributes");
  gtk_tree_view_append_column (GTK_TREE_VIEW (view), col);
  renderer = gtk_cell_renderer_text_new ();
  gtk_tree_view_column_pack_start (col, renderer, TRUE);
  gtk_tree_view_column_set_cell_data_func (col, renderer, cell_data_function_col_0_config_default, 0, 0);
  g_object_set (renderer, "editable", FALSE, (char*) 0);

  col = gtk_tree_view_column_new ();
  gtk_tree_view_column_set_title (col, "Attribute Value");
  gtk_tree_view_append_column (GTK_TREE_VIEW (view), col);
  renderer = gtk_cell_renderer_text_new ();
  g_signal_connect (renderer, "edited", (GCallback) cell_edited_callback_config_default, model);
  gtk_tree_view_column_pack_start (col, renderer, TRUE);
  gtk_tree_view_column_set_cell_data_func (col, renderer, cell_data_function_col_1_config_default, 0, 0);

  gtk_tree_view_set_model (GTK_TREE_VIEW (view), GTK_TREE_MODEL (model));

  g_object_unref (model); /* destroy model automatically with view */

  return view;
}

/*
 * Delete the tree model contents
 */
gboolean
clean_model_callback_config_default (GtkTreeModel *model, GtkTreePath *path,
                                     GtkTreeIter *iter, gpointer data)
{
  ModelTypeid *node = 0;
  gtk_tree_model_get (GTK_TREE_MODEL (model), iter, COL_TYPEID, &node, -1);
  if (node)
    {
      delete node;
    }
  gtk_tree_store_set (GTK_TREE_STORE (model), iter, COL_TYPEID, (ModelTypeid*) 0, -1);
  return FALSE;
}


} //end ns3 namespace

