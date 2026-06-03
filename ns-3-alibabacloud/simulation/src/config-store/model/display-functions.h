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
#ifndef DISPLAY_FUNCTIONS_H
#define DISPLAY_FUNCTIONS_H

#include "model-node-creator.h"
#include "model-typeid-creator.h"
#include <gtk/gtk.h>

namespace ns3 {
/**
 * This function includes the name of the attribute or the editable value
 * in the second column 
 *
 * \param col Pointer to the GtkTreeViewColumn
 * \param renderer Pointer to the GtkCellRenderer
 * \param model Pointer to the GtkTreeModel
 * \param iter  Pointer to the GtkTreeIter
 * \param user_data Pointer to the data to be displayed (or modified)
 */
void
cell_data_function_col_1 (GtkTreeViewColumn *col, GtkCellRenderer *renderer,
                          GtkTreeModel *model, GtkTreeIter *iter, gpointer user_data);
/**
 * This function includes the name of the object, pointer, vector or vector item
 * in the first column
 * \param col Pointer to the GtkTreeViewColumn
 * \param renderer Pointer to the GtkCellRenderer
 * \param model Pointer to the GtkTreeModel
 * \param iter  Pointer to the GtkTreeIter
 * \param user_data Pointer to the data to be displayed (or modified)
 */
void
cell_data_function_col_0 (GtkTreeViewColumn *col, GtkCellRenderer *renderer, GtkTreeModel *model,
                          GtkTreeIter *iter, gpointer user_data);
/**
 * This is the callback called when the value of an attribute is changed
 *
 *  \param cell the changed cell
 *  \param path_string the path
 *  \param new_text the updated text in the cell
 *  \param user_data user data
 */
void
cell_edited_callback (GtkCellRendererText *cell, gchar *path_string,
                      gchar *new_text, gpointer user_data);
/**
 * This function gets the column number 0 or 1 from the mouse
 * click
 * \param col the column being clicked
 * \returns the column index
 */
int
get_col_number_from_tree_view_column (GtkTreeViewColumn *col);
/**
 * This function displays the tooltip for an object, pointer, vector
 * item or an attribute
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
cell_tooltip_callback (GtkWidget *widget, gint x, gint y, gboolean keyboard_tip, 
                       GtkTooltip *tooltip, gpointer user_data);
/**
 * This is the main view opening the widget, getting tooltips and drawing the 
 * tree of attributes...
 * \param model the GtkTreeStore model
 * \returns a GtkWidget on success
 */
GtkWidget *
create_view (GtkTreeStore *model);
/**
 * Exit the window when exit button is pressed
 * \param button the pressed button
 * \param user_data
 */
void
exit_clicked_callback (GtkButton *button, gpointer user_data);
/**
 * Exit the application
 * \param widget a pointer to the widget
 * \param event the event responsible for the aplication exit
 * \param user_data user data
 * \returns true on clean exit
 */
gboolean
delete_event_callback (GtkWidget *widget, GdkEvent *event, gpointer user_data);
/**
 * Delete the tree model contents
 * \param model the GtkTreeModel
 * \param path the GtkTreePath
 * \param iter a GtkTreeIter
 * \param data user data
 * \return true if an error occurred.
 */
gboolean
clean_model_callback (GtkTreeModel *model, GtkTreePath *path,
                      GtkTreeIter *iter, gpointer data);


//     display functions used by default configurator

/**
 * This function writes data in the second column, this data is going to be editable
 * if it is a NODE_ATTRIBUTE
 *
 * \param col Pointer to the GtkTreeViewColumn
 * \param renderer Pointer to the GtkCellRenderer
 * \param model Pointer to the GtkTreeModel
 * \param iter  Pointer to the GtkTreeIter
 * \param user_data Pointer to the data to be displayed (or modified)
 */
void
cell_data_function_col_1_config_default (GtkTreeViewColumn *col, GtkCellRenderer *renderer, 
                                         GtkTreeModel *model, GtkTreeIter *iter,
                                         gpointer user_data);
/**
 * This function writes the attribute or typeid name in the column 0
 * \param col Pointer to the GtkTreeViewColumn
 * \param renderer Pointer to the GtkCellRenderer
 * \param model Pointer to the GtkTreeModel
 * \param iter  Pointer to the GtkTreeIter
 * \param user_data Pointer to the data to be displayed (or modified)
 */
void
cell_data_function_col_0_config_default (GtkTreeViewColumn *col, GtkCellRenderer *renderer, GtkTreeModel *model, 
                                         GtkTreeIter *iter, gpointer user_data);
/**
 * This is the action done when the user presses on the save button for the Default attributes.
 * It will save the config to a file.
 *
 * \param button (unused)
 * \param user_data
 */
void
save_clicked_default (GtkButton *button, gpointer user_data);
/**
 * If the user presses the button load, it will load the config file into memory for the Default attributes.
 *
 * \param button (unused)
 * \param user_data
 */
void
load_clicked_default (GtkButton *button, gpointer user_data);
/**
 * This is the action done when the user presses on the save button for the Attributes.
 * It will save the config to a file.
 *
 * \param button (unused)
 * \param user_data
 */
void
save_clicked_attribute (GtkButton *button, gpointer user_data);
/**
 * If the user presses the button load, it will load the config file into memory for the Attributes.
 *
 * \param button (unused)
 * \param user_data
 */
void
load_clicked_attribute (GtkButton *button, gpointer user_data);
/**
 *  This functions is called whenever there is a change in the value of an attribute
 *  If the input value is ok, it will be updated in the default value and in the
 *  GUI, otherwise, it won't be updated in both.
 *
 *  \param cell the changed cell
 *  \param path_string the path
 *  \param new_text the updated text in the cell
 *  \param user_data user data
 */
void
cell_edited_callback_config_default (GtkCellRendererText *cell, gchar *path_string,
                                     gchar *new_text, gpointer user_data);
/**
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
                                      gboolean keyboard_tip, GtkTooltip *tooltip, gpointer user_data);
/**
 * This is the main view opening the widget, getting tooltips and drawing the 
 * tree of attributes
 * \param model the GtkTreeStore model
 * \returns a GtkWidget on success
 */
GtkWidget *
create_view_config_default (GtkTreeStore *model);
/**
 * Delete the tree model contents
 * \param model the GtkTreeModel
 * \param path the GtkTreePath
 * \param iter a GtkTreeIter
 * \param data user data
 * \return true if an error occurred.
 */
gboolean
clean_model_callback_config_default (GtkTreeModel *model, GtkTreePath *path,
                                     GtkTreeIter *iter, gpointer data);
} //end namespace ns3

#endif
