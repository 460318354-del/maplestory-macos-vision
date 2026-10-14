from __future__ import annotations

from typing import Callable

import AppKit
import objc


class FrameSelectionCanvas(AppKit.NSView):
    """Displays a captured game frame and supports band/rectangle selection."""

    def initWithFrame_(self, frame):
        self = objc.super(FrameSelectionCanvas, self).initWithFrame_(frame)
        if self is None:
            return None
        self.image = None
        self.image_size = (0, 0)
        self.selection_y = None
        self.selecting = False
        self.selection_enabled = False
        self.on_selection: Callable[[tuple[int, int]], None] | None = None
        self.rectangle_enabled = False
        self.on_rectangle: Callable[[tuple[int, int, int, int]], None] | None = None
        self.click_enabled = False
        self.click_point = None
        self.on_click: Callable[[tuple[int, int]], None] | None = None
        self.selection_mode = "band"
        self.selection_rect = None
        self.detection_rect = None
        self.region_rect = None
        # These are game-frame coordinates. The two markers intentionally
        # remain separate because patrol and fixed-point combat are independent
        # workflows, even when both points are placed inside the minimap.
        self.patrol_marker_point = None
        self.fixed_marker_point = None
        self.minimap_rect = None
        self.route_regions = ()
        self.ladder_markers = ()
        self.magnifier_enabled = False
        self.magnifier_image = None
        self.magnifier_size = (0, 0)
        self.magnifier_point = None
        self.magnifier_cursor = None
        self.on_mouse_move: Callable[[tuple[int, int]], None] | None = None
        return self

    def setImage_pixelSize_(self, image, pixel_size):
        self.image = image
        self.image_size = (int(pixel_size[0]), int(pixel_size[1]))
        self.selection_y = None
        self.selection_rect = None
        self.detection_rect = None
        self.region_rect = None
        self.magnifier_image = None
        self.magnifier_point = None
        self.magnifier_cursor = None
        self.setNeedsDisplay_(True)

    def setMagnifierEnabled_(self, enabled):
        self.magnifier_enabled = bool(enabled)
        if not self.magnifier_enabled:
            self.magnifier_image = None
            self.magnifier_point = None
        self.setNeedsDisplay_(True)

    def setMagnifierImage_pixelSize_(self, image, pixel_size):
        self.magnifier_image = image
        self.magnifier_size = (
            int(pixel_size[0]),
            int(pixel_size[1]),
        )
        self.setNeedsDisplay_(True)

    def setMagnifierCursor_(self, point):
        self.magnifier_cursor = (
            int(point[0]),
            int(point[1]),
        ) if point is not None else None
        self.setNeedsDisplay_(True)

    def setSelectionEnabled_(self, enabled):
        self.selection_enabled = bool(enabled)
        self.selecting = False
        self.setNeedsDisplay_(True)

    def setRectangleEnabled_(self, enabled):
        self.rectangle_enabled = bool(enabled)
        self.selecting = False
        self.setNeedsDisplay_(True)

    def setClickEnabled_(self, enabled):
        self.click_enabled = bool(enabled)
        self.click_point = None
        self.selecting = False
        self.setNeedsDisplay_(True)

    def setSelectionMode_(self, mode):
        self.selection_mode = str(mode)
        self.selection_y = None
        self.selection_rect = None
        self.selecting = False
        self.setNeedsDisplay_(True)

    def setDetectionRect_(self, detection):
        self.detection_rect = detection
        self.setNeedsDisplay_(True)

    def setRegionRect_(self, region):
        """Persistently draw the configured monster-search ROI."""
        self.region_rect = region
        self.setNeedsDisplay_(True)

    def setMarkerPoint_(self, point):
        """Backward-compatible alias for the patrol marker."""
        self.setPatrolMarkerPoint_(point)

    def setPatrolMarkerPoint_(self, point):
        self.patrol_marker_point = point
        self.setNeedsDisplay_(True)

    def setFixedMarkerPoint_(self, point):
        self.fixed_marker_point = point
        self.setNeedsDisplay_(True)

    def setMinimapRect_(self, rectangle):
        self.minimap_rect = rectangle
        self.setNeedsDisplay_(True)

    def setRouteRegions_(self, regions):
        self.route_regions = tuple(regions or ())
        self.setNeedsDisplay_(True)

    def setLadderMarkers_(self, markers):
        self.ladder_markers = tuple(markers or ())
        self.setNeedsDisplay_(True)

    def setSelectionY_(self, selection):
        self.selection_y = selection
        self.setNeedsDisplay_(True)

    def isFlipped(self):
        return True

    def updateTrackingAreas(self):
        for area in self.trackingAreas():
            self.removeTrackingArea_(area)
        options = (
            AppKit.NSTrackingMouseMoved
            | AppKit.NSTrackingActiveInKeyWindow
            | AppKit.NSTrackingInVisibleRect
        )
        area = AppKit.NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            self.bounds(),
            options,
            self,
            None,
        )
        self.addTrackingArea_(area)
        objc.super(FrameSelectionCanvas, self).updateTrackingAreas()

    def mouseMoved_(self, event):
        if not self.magnifier_enabled or self.on_mouse_move is None:
            return
        point = self.imagePointForEvent(event)
        if point is not None:
            self.magnifier_point = point
            self.on_mouse_move(point)
            self.setNeedsDisplay_(True)

    @objc.python_method
    def imageDisplayRect(self):
        if self.image is None or not self.image_size[0] or not self.image_size[1]:
            return AppKit.NSZeroRect
        bounds = self.bounds()
        image_width, image_height = self.image_size
        scale = min(
            bounds.size.width / image_width,
            bounds.size.height / image_height,
        )
        width = image_width * scale
        height = image_height * scale
        return AppKit.NSMakeRect(
            bounds.origin.x + (bounds.size.width - width) / 2,
            bounds.origin.y + (bounds.size.height - height) / 2,
            width,
            height,
        )

    @objc.python_method
    def imageYForEvent(self, event):
        display_rect = self.imageDisplayRect()
        if display_rect.size.height <= 0:
            return None
        point = self.convertPoint_fromView_(event.locationInWindow(), None)
        relative_y = (point.y - display_rect.origin.y) / display_rect.size.height
        relative_y = max(0.0, min(1.0, relative_y))
        return round(relative_y * self.image_size[1])

    @objc.python_method
    def imagePointForEvent(self, event):
        display_rect = self.imageDisplayRect()
        if display_rect.size.width <= 0 or display_rect.size.height <= 0:
            return None
        point = self.convertPoint_fromView_(event.locationInWindow(), None)
        relative_x = (point.x - display_rect.origin.x) / display_rect.size.width
        relative_y = (point.y - display_rect.origin.y) / display_rect.size.height
        relative_x = max(0.0, min(1.0, relative_x))
        relative_y = max(0.0, min(1.0, relative_y))
        return (
            round(relative_x * self.image_size[0]),
            round(relative_y * self.image_size[1]),
        )

    def drawRect_(self, rect):
        AppKit.NSColor.whiteColor().set()
        AppKit.NSRectFill(self.bounds())
        if self.image is None:
            return

        display_rect = self.imageDisplayRect()
        self.image.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
            display_rect,
            AppKit.NSZeroRect,
            AppKit.NSCompositingOperationSourceOver,
            1.0,
            True,
            None,
        )

        image_height = self.image_size[1]
        image_width = self.image_size[0]
        scale_x = display_rect.size.width / image_width
        scale_y = display_rect.size.height / image_height
        if self.selection_y is not None and self.selection_mode != "rect":
            y0, y1 = self.selection_y
            top = display_rect.origin.y + min(y0, y1) * scale_y
            bottom = display_rect.origin.y + max(y0, y1) * scale_y
            band = AppKit.NSMakeRect(
                display_rect.origin.x,
                top,
                display_rect.size.width,
                max(1.0, bottom - top),
            )
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.1, 0.45, 1.0, 0.20
            ).set()
            AppKit.NSRectFillUsingOperation(
                band,
                AppKit.NSCompositingOperationSourceOver,
            )
            AppKit.NSColor.systemBlueColor().set()
            path = AppKit.NSBezierPath.bezierPath()
            path.moveToPoint_(AppKit.NSMakePoint(display_rect.origin.x, top))
            path.lineToPoint_(
                AppKit.NSMakePoint(
                    display_rect.origin.x + display_rect.size.width,
                    top,
                )
            )
            path.moveToPoint_(AppKit.NSMakePoint(display_rect.origin.x, bottom))
            path.lineToPoint_(
                AppKit.NSMakePoint(
                    display_rect.origin.x + display_rect.size.width,
                    bottom,
                )
            )
            path.setLineWidth_(2.0)
            path.stroke()

        if self.selection_rect is not None and self.selection_mode == "rect":
            x0, y0, x1, y1 = self.selection_rect
            rect = AppKit.NSMakeRect(
                display_rect.origin.x + min(x0, x1) * scale_x,
                display_rect.origin.y + min(y0, y1) * scale_y,
                max(1.0, abs(x1 - x0) * scale_x),
                max(1.0, abs(y1 - y0) * scale_y),
            )
            AppKit.NSColor.systemRedColor().set()
            path = AppKit.NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(2.0)
            path.stroke()

        if self.detection_rect is not None:
            x, y, width, height = self.detection_rect
            rect = AppKit.NSMakeRect(
                display_rect.origin.x + x * scale_x,
                display_rect.origin.y + y * scale_y,
                max(1.0, width * scale_x),
                max(1.0, height * scale_y),
            )
            AppKit.NSColor.systemGreenColor().set()
            path = AppKit.NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(3.0)
            path.stroke()

        if self.region_rect is not None:
            x, y, width, height = self.region_rect
            region = AppKit.NSMakeRect(
                display_rect.origin.x + x * scale_x,
                display_rect.origin.y + y * scale_y,
                max(1.0, width * scale_x),
                max(1.0, height * scale_y),
            )
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                1.0, 0.65, 0.0, 0.12
            ).set()
            AppKit.NSRectFillUsingOperation(
                region,
                AppKit.NSCompositingOperationSourceOver,
            )
            AppKit.NSColor.systemOrangeColor().set()
            path = AppKit.NSBezierPath.bezierPathWithRect_(region)
            path.setLineWidth_(2.5)
            path.stroke()

        if self.minimap_rect is not None:
            x, y, width, height = self.minimap_rect
            rect = AppKit.NSMakeRect(
                display_rect.origin.x + x * scale_x,
                display_rect.origin.y + y * scale_y,
                max(1.0, width * scale_x),
                max(1.0, height * scale_y),
            )
            AppKit.NSColor.systemGreenColor().set()
            path = AppKit.NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(2.0)
            path.stroke()

        for name, rectangle in self.route_regions:
            x, y, width, height = rectangle
            rect = AppKit.NSMakeRect(
                display_rect.origin.x + x * scale_x,
                display_rect.origin.y + y * scale_y,
                max(1.0, width * scale_x),
                max(1.0, height * scale_y),
            )
            AppKit.NSColor.systemOrangeColor().set()
            path = AppKit.NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(2.0)
            path.stroke()
            self._draw_text(
                name,
                rect.origin.x,
                rect.origin.y + rect.size.height + 2,
                AppKit.NSColor.systemOrangeColor(),
            )

        for name, point in self.ladder_markers:
            self._draw_small_point(
                display_rect,
                scale_x,
                scale_y,
                point,
                AppKit.NSColor.systemPurpleColor(),
            )
            x, y = point
            self._draw_text(
                name,
                display_rect.origin.x + x * scale_x + 6,
                display_rect.origin.y + y * scale_y + 4,
                AppKit.NSColor.systemPurpleColor(),
            )

        self._draw_marker(
            display_rect,
            scale_x,
            scale_y,
            self.patrol_marker_point,
            AppKit.NSColor.systemOrangeColor(),
        )
        self._draw_marker(
            display_rect,
            scale_x,
            scale_y,
            self.fixed_marker_point,
            AppKit.NSColor.systemPurpleColor(),
        )
        self._draw_magnifier()

    def _draw_magnifier(self) -> None:
        if (
            not self.magnifier_enabled
            or self.magnifier_image is None
            or self.magnifier_point is None
        ):
            return
        bounds = self.bounds()
        width = min(220.0, max(120.0, bounds.size.width * 0.32))
        height = min(150.0, max(90.0, bounds.size.height * 0.32))
        rect = AppKit.NSMakeRect(
            max(8.0, bounds.size.width - width - 10.0),
            max(8.0, bounds.size.height - height - 10.0),
            width,
            height,
        )
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.05, 0.92).set()
        AppKit.NSRectFill(rect)
        self.magnifier_image.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
            rect,
            AppKit.NSZeroRect,
            AppKit.NSCompositingOperationSourceOver,
            1.0,
            True,
            None,
        )
        AppKit.NSColor.systemPurpleColor().set()
        border = AppKit.NSBezierPath.bezierPathWithRect_(rect)
        border.setLineWidth_(2.0)
        border.stroke()
        cursor_x, cursor_y = self.magnifier_cursor or (
            self.magnifier_size[0] / 2.0,
            self.magnifier_size[1] / 2.0,
        )
        source_width = max(1, self.magnifier_size[0])
        source_height = max(1, self.magnifier_size[1])
        cx = rect.origin.x + cursor_x * rect.size.width / source_width
        cy = rect.origin.y + cursor_y * rect.size.height / source_height
        cross = AppKit.NSBezierPath.bezierPath()
        cross.moveToPoint_(AppKit.NSMakePoint(cx - 10, cy))
        cross.lineToPoint_(AppKit.NSMakePoint(cx + 10, cy))
        cross.moveToPoint_(AppKit.NSMakePoint(cx, cy - 10))
        cross.lineToPoint_(AppKit.NSMakePoint(cx, cy + 10))
        cross.setLineWidth_(2.0)
        cross.stroke()
        self._draw_text(
            f"({self.magnifier_point[0]},{self.magnifier_point[1]})",
            rect.origin.x + 6,
            rect.origin.y + 6,
            AppKit.NSColor.systemPurpleColor(),
        )

    @staticmethod
    def _draw_marker(display_rect, scale_x, scale_y, point, color) -> None:
        if point is None:
            return
        x, y = point
        cx = display_rect.origin.x + x * scale_x
        cy = display_rect.origin.y + y * scale_y
        radius = 5.0
        color.set()
        marker = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            AppKit.NSMakeRect(
                cx - radius,
                cy - radius,
                radius * 2,
                radius * 2,
            )
        )
        marker.setLineWidth_(2.0)
        marker.stroke()
        path = AppKit.NSBezierPath.bezierPath()
        path.moveToPoint_(AppKit.NSMakePoint(cx - 8, cy))
        path.lineToPoint_(AppKit.NSMakePoint(cx + 8, cy))
        path.moveToPoint_(AppKit.NSMakePoint(cx, cy - 8))
        path.lineToPoint_(AppKit.NSMakePoint(cx, cy + 8))
        path.setLineWidth_(2.0)
        path.stroke()

    @staticmethod
    def _draw_small_point(display_rect, scale_x, scale_y, point, color) -> None:
        """Draw a compact route marker that remains readable when points cluster."""
        if point is None:
            return
        x, y = point
        cx = display_rect.origin.x + x * scale_x
        cy = display_rect.origin.y + y * scale_y
        radius = 2.5
        color.set()
        marker = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            AppKit.NSMakeRect(
                cx - radius,
                cy - radius,
                radius * 2,
                radius * 2,
            )
        )
        marker.fill()

    @staticmethod
    def _draw_text(text, x, y, color) -> None:
        attributes = {
            AppKit.NSForegroundColorAttributeName: color,
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(11),
        }
        NSString = getattr(AppKit, "NSString", None)
        if NSString is not None:
            NSString.stringWithString_(str(text)).drawAtPoint_withAttributes_(
                AppKit.NSMakePoint(x, y),
                attributes,
            )

    def mouseDown_(self, event):
        if (
            (
                not self.selection_enabled
                and not self.rectangle_enabled
                and not self.click_enabled
            )
            or self.image is None
        ):
            return
        point = self.imagePointForEvent(event)
        if point is None:
            return
        if self.click_enabled and not self.rectangle_enabled and not self.selection_enabled:
            self.clicking = True
            self.click_point = point
            return
        self.selecting = True
        if self.rectangle_enabled:
            self.selection_rect = (point[0], point[1], point[0], point[1])
        else:
            self.selection_y = (point[1], point[1])
        self.setNeedsDisplay_(True)

    def mouseDragged_(self, event):
        if getattr(self, "clicking", False):
            return
        if not self.selecting:
            return
        point = self.imagePointForEvent(event)
        if point is None:
            return
        if self.rectangle_enabled:
            x0, y0, _, _ = self.selection_rect
            self.selection_rect = (x0, y0, point[0], point[1])
        else:
            self.selection_y = (self.selection_y[0], point[1])
        self.setNeedsDisplay_(True)

    def mouseUp_(self, event):
        if getattr(self, "clicking", False):
            self.clicking = False
            point = self.imagePointForEvent(event) or self.click_point
            self.click_point = None
            if point is not None and self.on_click is not None:
                self.on_click(point)
            return
        if not self.selecting:
            return
        self.selecting = False
        point = self.imagePointForEvent(event)
        if point is not None:
            if self.rectangle_enabled:
                x0, y0, _, _ = self.selection_rect
                self.selection_rect = (x0, y0, point[0], point[1])
            else:
                self.selection_y = (self.selection_y[0], point[1])
        if self.rectangle_enabled:
            if hasattr(self, "selection_rect") and self.on_rectangle is not None:
                x0, y0, x1, y1 = self.selection_rect
                self.on_rectangle(
                    (
                        min(x0, x1),
                        min(y0, y1),
                        abs(x1 - x0),
                        abs(y1 - y0),
                    )
                )
        elif self.selection_y is not None and self.on_selection is not None:
            self.on_selection(tuple(sorted(self.selection_y)))
        self.setNeedsDisplay_(True)
